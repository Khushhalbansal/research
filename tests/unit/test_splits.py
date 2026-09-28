import hashlib

from lrmc.data.splits import (
    Fold,
    SampleRecord,
    assert_fold_disjoint,
    assert_group_disjoint,
    group_aware_holdout,
    leave_one_family_out,
    random_k_unknown,
)


def _mk_records(n_families=6, n_per_family=20, n_dup_per_family=2, seed=0) -> list[SampleRecord]:
    records = []
    for fam_idx in range(n_families):
        family = f"Fam{fam_idx}"
        first_hash = None
        for i in range(n_per_family):
            content = f"{family}-{i}-{seed}".encode()
            sha = hashlib.sha256(content).hexdigest()
            if i == 0:
                first_hash = sha
            records.append(SampleRecord(sample_id=f"{family}_{i}", family=family, sha256=sha))
        # inject duplicates: same sha256 as sample 0, different sample_id/filename
        for d in range(n_dup_per_family):
            records.append(
                SampleRecord(sample_id=f"{family}_dup{d}", family=family, sha256=first_hash)
            )
    return records


def test_leave_one_family_out_covers_every_family():
    records = _mk_records()
    folds = leave_one_family_out(records, val_frac=0.15, test_frac=0.2, seed=0)
    families = sorted({r.family for r in records})
    assert len(folds) == len(families)
    unknowns = sorted(f.unknown_families[0] for f in folds)
    assert unknowns == families


def test_leave_one_family_out_disjoint_and_dedup_safe():
    records = _mk_records()
    folds = leave_one_family_out(records, seed=1)
    for fold in folds:
        assert_fold_disjoint(fold, records)


def test_random_k_unknown_disjoint():
    records = _mk_records()
    fold = random_k_unknown(records, n_unknown=2, seed=42)
    assert len(fold.unknown_families) == 2
    assert_fold_disjoint(fold, records)
    # known + unknown families partition the full family set
    all_families = {r.family for r in records}
    assert set(fold.known_families) | set(fold.unknown_families) == all_families


def test_random_k_unknown_seeded_reproducible():
    records = _mk_records()
    fold_a = random_k_unknown(records, n_unknown=2, seed=7)
    fold_b = random_k_unknown(records, n_unknown=2, seed=7)
    assert fold_a.unknown_families == fold_b.unknown_families
    assert fold_a.train == fold_b.train


def test_random_k_unknown_different_seeds_can_differ():
    records = _mk_records(n_families=10)
    seeds_unknowns = set()
    for s in range(6):
        fold = random_k_unknown(records, n_unknown=2, seed=s)
        seeds_unknowns.add(tuple(fold.unknown_families))
    assert len(seeds_unknowns) > 1


def test_pseudo_unknown_removed_from_known_splits():
    records = _mk_records()
    fold = random_k_unknown(records, n_unknown=1, seed=3, pseudo_unknown=True)
    assert fold.pseudo_unknown_family is not None
    assert fold.pseudo_unknown_family not in fold.known_families
    assert len(fold.pseudo_unknown_val) > 0
    assert_fold_disjoint(fold, records)


def test_group_aware_holdout_never_splits_a_group():
    records = _mk_records(n_families=8)
    family_to_group = {
        "Fam0": "GroupA",
        "Fam1": "GroupA",
        "Fam2": "GroupB",
        "Fam3": "GroupB",
    }
    for seed in range(10):
        fold = group_aware_holdout(
            records, family_to_group, n_unknown_groups=2, seed=seed
        )
        assert_group_disjoint(fold, family_to_group)
        assert_fold_disjoint(fold, records)
        # if GroupA's Fam0 is unknown, Fam1 must be too (and vice versa)
        if "Fam0" in fold.unknown_families:
            assert "Fam1" in fold.unknown_families
        if "Fam0" in fold.known_families:
            assert "Fam1" in fold.known_families


def test_naive_random_k_unknown_can_split_a_group_by_construction():
    """Documents the known leakage risk of the naive protocol (vs. group-aware)."""
    records = _mk_records(n_families=8)
    family_to_group = {"Fam0": "GroupA", "Fam1": "GroupA"}
    split_found = False
    for seed in range(30):
        fold = random_k_unknown(records, n_unknown=3, seed=seed)
        fam0_unknown = "Fam0" in fold.unknown_families
        fam1_unknown = "Fam1" in fold.unknown_families
        if fam0_unknown != fam1_unknown:
            split_found = True
            break
    assert split_found, "expected naive protocol to eventually split a variant group"


def test_stratified_split_no_duplicate_sample_ids_within_fold():
    records = _mk_records()
    fold = random_k_unknown(records, n_unknown=1, seed=5)
    all_ids = fold.train + fold.val_known + fold.test_known + fold.test_unknown
    assert len(all_ids) == len(set(all_ids))


def test_manifest_roundtrip(tmp_path):
    records = _mk_records()
    fold = random_k_unknown(records, n_unknown=1, seed=9)
    path = tmp_path / "fold.json"
    fold.to_json(path)
    loaded = Fold.from_json(path)
    assert loaded == fold
