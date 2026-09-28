"""Shared, leakage-tested split protocols for both Malimg and BIG 2015.

A dataset loader turns files on disk into a flat list of :class:`SampleRecord`
(``sample_id``, ``family``, ``sha256``, optional ``group``). Everything downstream
-- training, baselines, evaluation -- consumes a :class:`Fold`, which names five
disjoint sample-id buckets: ``train``, ``val_known``, ``test_known``,
``test_unknown`` and an optional ``pseudo_unknown_val`` (a known family withheld
from train/val/test-known and used only for threshold/early-stopping calibration
without ever touching a true unknown family or the test-unknown set).

Three protocols are supported:

* :func:`leave_one_family_out` -- one fold per known family, that family held
  out as unknown (family-level; ignores variant groups).
* :func:`random_k_unknown` -- seeded random choice of ``k`` families as unknown,
  **naively** (a group's variants can be split between known and unknown --
  this is a deliberate leakage risk, kept so its effect can be measured against
  the group-aware protocol below).
* :func:`group_aware_holdout` -- seeded random choice of ``k`` *groups* (a family
  with no configured group is its own singleton group) as unknown, so closely
  related variant families are never split across the known/unknown boundary.

Every hash-group (samples sharing an identical sha256) is assigned to exactly one
of train/val/test as a unit, so exact duplicates never straddle a split -- this
matters most for Malimg, which the mission notes has "many exact duplicates".
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class SampleRecord:
    sample_id: str
    family: str
    sha256: str
    group: str | None = None
    path: str = ""


@dataclass
class Fold:
    protocol: str
    seed: int
    known_families: list[str]
    unknown_families: list[str]
    train: list[str]
    val_known: list[str]
    test_known: list[str]
    test_unknown: list[str]
    pseudo_unknown_val: list[str] = field(default_factory=list)
    pseudo_unknown_family: str | None = None
    fold_index: int = 0

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, sort_keys=True)

    @staticmethod
    def from_json(path: str | Path) -> Fold:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return Fold(**data)


def load_family_groups(path: str | Path) -> dict[str, str]:
    """Load a ``groups:`` YAML (see configs/data/malimg_groups.yaml) into a flat
    ``family -> group_name`` map for :func:`group_aware_holdout`."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    family_to_group: dict[str, str] = {}
    for group_name, members in (raw.get("groups") or {}).items():
        for fam in members:
            family_to_group[fam] = group_name
    return family_to_group


# ---------------------------------------------------------------------------
# Hash-aware stratified split of the KNOWN portion into train/val/test
# ---------------------------------------------------------------------------


def _hash_groups(records: list[SampleRecord]) -> dict[str, list[SampleRecord]]:
    groups: dict[str, list[SampleRecord]] = defaultdict(list)
    for r in records:
        groups[r.sha256].append(r)
    return groups


def stratified_hash_split(
    records: list[SampleRecord],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Split KNOWN records into (train_ids, val_ids, test_ids).

    Splits at the hash-group level (never breaking an exact duplicate across
    sets) and stratifies by family so every known family with >=1 hash-group is
    represented in train wherever possible.
    """
    import random as _random

    rng = _random.Random(seed)
    hash_groups = _hash_groups(records)

    # majority family per hash-group (duplicates should share a family; if not,
    # fall back to the most common label in the group rather than crashing).
    family_of_group: dict[str, str] = {}
    for h, recs in hash_groups.items():
        counts = Counter(r.family for r in recs)
        family_of_group[h] = counts.most_common(1)[0][0]

    groups_by_family: dict[str, list[str]] = defaultdict(list)
    for h, fam in family_of_group.items():
        groups_by_family[fam].append(h)

    train_hashes: list[str] = []
    val_hashes: list[str] = []
    test_hashes: list[str] = []

    for _fam, hashes in groups_by_family.items():
        hashes = sorted(hashes)
        rng.shuffle(hashes)
        n = len(hashes)
        n_val = round(n * val_frac)
        n_test = round(n * test_frac)
        # keep at least one hash-group in train if any exist, capping val+test
        if n_val + n_test >= n and n > 0:
            n_val = min(n_val, max(0, n - 1))
            n_test = min(n_test, max(0, n - n_val - (1 if n - n_val > 0 else 0)))
        val_hashes.extend(hashes[:n_val])
        test_hashes.extend(hashes[n_val : n_val + n_test])
        train_hashes.extend(hashes[n_val + n_test :])

    def _ids(hashes: list[str]) -> list[str]:
        out = []
        for h in hashes:
            out.extend(r.sample_id for r in hash_groups[h])
        return out

    return _ids(train_hashes), _ids(val_hashes), _ids(test_hashes)


def _carve_pseudo_unknown(
    known_records: list[SampleRecord], family: str | None, seed: int
) -> tuple[str | None, list[SampleRecord], list[str]]:
    """Remove one known family entirely; return (family, remaining_records, ids removed)."""
    if not known_records:
        return None, known_records, []
    if family is None:
        import random as _random

        families = sorted({r.family for r in known_records})
        if not families:
            return None, known_records, []
        family = _random.Random(seed).choice(families)
    removed = [r for r in known_records if r.family == family]
    remaining = [r for r in known_records if r.family != family]
    return family, remaining, [r.sample_id for r in removed]


def _build_fold(
    protocol: str,
    seed: int,
    all_records: list[SampleRecord],
    unknown_families: set[str],
    val_frac: float,
    test_frac: float,
    pseudo_unknown: bool,
    pseudo_unknown_family: str | None,
    fold_index: int = 0,
) -> Fold:
    known_records = [r for r in all_records if r.family not in unknown_families]
    unknown_records = [r for r in all_records if r.family in unknown_families]

    pu_family = None
    pu_ids: list[str] = []
    if pseudo_unknown:
        pu_family, known_records, pu_ids = _carve_pseudo_unknown(
            known_records, pseudo_unknown_family, seed
        )

    train_ids, val_ids, test_ids = stratified_hash_split(known_records, val_frac, test_frac, seed)
    known_families = sorted({r.family for r in known_records})

    return Fold(
        protocol=protocol,
        seed=seed,
        known_families=known_families,
        unknown_families=sorted(unknown_families),
        train=train_ids,
        val_known=val_ids,
        test_known=test_ids,
        test_unknown=[r.sample_id for r in unknown_records],
        pseudo_unknown_val=pu_ids,
        pseudo_unknown_family=pu_family,
        fold_index=fold_index,
    )


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


def leave_one_family_out(
    records: list[SampleRecord],
    val_frac: float = 0.15,
    test_frac: float = 0.2,
    seed: int = 0,
    pseudo_unknown: bool = False,
) -> list[Fold]:
    """One fold per family: that family is unknown, all others are known."""
    families = sorted({r.family for r in records})
    folds = []
    for i, fam in enumerate(families):
        fold = _build_fold(
            protocol="leave_one_family_out",
            seed=seed,
            all_records=records,
            unknown_families={fam},
            val_frac=val_frac,
            test_frac=test_frac,
            pseudo_unknown=pseudo_unknown,
            pseudo_unknown_family=None,
            fold_index=i,
        )
        folds.append(fold)
    return folds


def random_k_unknown(
    records: list[SampleRecord],
    n_unknown: int,
    seed: int = 0,
    val_frac: float = 0.15,
    test_frac: float = 0.2,
    pseudo_unknown: bool = False,
) -> Fold:
    """Naive random choice of k unknown families (may split a variant group)."""
    import random as _random

    families = sorted({r.family for r in records})
    if n_unknown >= len(families):
        raise ValueError("n_unknown must leave at least one known family")
    rng = _random.Random(seed)
    unknown = set(rng.sample(families, n_unknown))
    return _build_fold(
        protocol="random_k_unknown",
        seed=seed,
        all_records=records,
        unknown_families=unknown,
        val_frac=val_frac,
        test_frac=test_frac,
        pseudo_unknown=pseudo_unknown,
        pseudo_unknown_family=None,
    )


def group_aware_holdout(
    records: list[SampleRecord],
    family_to_group: dict[str, str],
    n_unknown_groups: int,
    seed: int = 0,
    val_frac: float = 0.15,
    test_frac: float = 0.2,
    pseudo_unknown: bool = False,
) -> Fold:
    """Random choice of k whole groups as unknown; a family with no entry in
    ``family_to_group`` is treated as its own singleton group, so it can still
    be chosen but never splits a real variant family apart."""
    import random as _random

    families = sorted({r.family for r in records})
    group_of = {fam: family_to_group.get(fam, f"__singleton__{fam}") for fam in families}
    groups: dict[str, list[str]] = defaultdict(list)
    for fam, g in group_of.items():
        groups[g].append(fam)

    group_names = sorted(groups.keys())
    if n_unknown_groups >= len(group_names):
        raise ValueError("n_unknown_groups must leave at least one known group")
    rng = _random.Random(seed)
    unknown_groups = set(rng.sample(group_names, n_unknown_groups))
    unknown_families = {fam for g in unknown_groups for fam in groups[g]}

    fold = _build_fold(
        protocol="group_aware_holdout",
        seed=seed,
        all_records=records,
        unknown_families=unknown_families,
        val_frac=val_frac,
        test_frac=test_frac,
        pseudo_unknown=pseudo_unknown,
        pseudo_unknown_family=None,
    )
    assert_group_disjoint(fold, family_to_group)
    return fold


# ---------------------------------------------------------------------------
# Leakage assertions
# ---------------------------------------------------------------------------


def assert_fold_disjoint(fold: Fold, records: list[SampleRecord]) -> None:
    """Sample-id AND sha256 disjointness across all five buckets."""
    buckets = {
        "train": fold.train,
        "val_known": fold.val_known,
        "test_known": fold.test_known,
        "test_unknown": fold.test_unknown,
        "pseudo_unknown_val": fold.pseudo_unknown_val,
    }
    id_to_hash = {r.sample_id: r.sha256 for r in records}

    seen_ids: set[str] = set()
    for name, ids in buckets.items():
        id_set = set(ids)
        assert len(id_set) == len(ids), f"duplicate sample_id within bucket {name}"
        overlap = seen_ids & id_set
        assert not overlap, f"sample_id overlap involving bucket {name}: {overlap}"
        seen_ids |= id_set

    seen_hashes: dict[str, str] = {}
    for name, ids in buckets.items():
        for sid in ids:
            h = id_to_hash[sid]
            if h in seen_hashes and seen_hashes[h] != name:
                raise AssertionError(
                    f"sha256 {h} appears in both bucket {seen_hashes[h]} and {name} "
                    "(duplicate straddling a split)"
                )
            seen_hashes[h] = name


def assert_group_disjoint(fold: Fold, family_to_group: dict[str, str]) -> None:
    """For group-aware folds: no group may have members in both known and unknown."""
    known = set(fold.known_families) | (
        {fold.pseudo_unknown_family} if fold.pseudo_unknown_family else set()
    )
    unknown = set(fold.unknown_families)
    known_groups = {family_to_group.get(f, f"__singleton__{f}") for f in known}
    unknown_groups = {family_to_group.get(f, f"__singleton__{f}") for f in unknown}
    overlap = known_groups & unknown_groups
    assert not overlap, f"group(s) {overlap} split across known/unknown"
