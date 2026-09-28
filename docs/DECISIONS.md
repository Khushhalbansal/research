# Decisions log

One-line-rationale record of choices made under ambiguity, per the mission's principle
5 ("choose the most defensible option, document it, continue"). Newest at bottom.

- **LRMC loss form follows the MISSION spec, not the earlier markdown draft.**
  `/LRMC_Zero-Day_Malware_Paper.md` uses a single combined `L_rad` with `m_in`/`m_out`
  margins and a separate volume penalty `L_vol`; the mission's task text specifies
  `L = L_supcon + alpha*L_in + beta*L_out + gamma*L_rad` with `L_rad` as a pure
  tightness term (`mean_c r_c`). The mission text is the authoritative, explicit
  instruction for this build ("implement exactly"), so it wins; `docs/LRMC.md`
  documents the mission's version and notes the draft as a related but superseded
  formulation so the paper's Methodology section can be reconciled by hand.
- **CLI uses stdlib `argparse`**, not `click`/`typer`, to avoid adding a dependency that
  might be unavailable offline on Kaggle; the whole tool surface (`train`, `evaluate`,
  `infer`, `scan`, `run-queue`, `aggregate`) is a handful of subcommands, well within
  argparse's comfort zone.
- **Config format is YAML with dataclass schemas** (not pure YAML dicts) so config
  errors are caught at load time (fail fast) and config hashing is stable
  (canonical JSON dump of the dataclass, sorted keys).
- **Config hash = sha256 of canonical JSON of the fully-resolved config** (after
  defaults are applied), stored in every `metrics.json`; this makes two runs
  comparable iff their *effective* config matches, not just their YAML file.
- **`tiny_test` ViT preset**: img 64, patch 8, depth 2, embed dim 64, heads 2, mlp
  ratio 2 — small enough that a full smoke train+eval completes in well under a
  minute on CPU, per the mission's CPU-rehearsal requirement.
- **Pretrained-weight loader fallback order**: (1) `timm` pretrained download if
  network reachable, (2) local weights file path from config
  (`model.local_weights_path`), (3) random init with a `logging.warning` and
  `pretrained_source: "random_init"` recorded in `metrics.json`. Chosen because
  Kaggle may run with internet off and the mission requires this to be robust and
  logged, not silently wrong.
- **DataParallel wraps only backbone+head** (per mission instruction) — a code
  comment in `engine/train.py` explains why: SupCon needs the *full* batch of
  negatives; per-GPU-replica loss computation would silently use only that
  replica's shard as negatives, weakening the contrastive signal. The loss itself
  (and prototype/radii update) runs on the gathered full-batch embeddings on the
  primary device.
- **Prototype EMA vs. batch-mean is a config switch** (`prototypes.mode: "ema" |
  "batch_mean"`), default `ema` per mission; both are unit-tested including the
  missing-class-in-batch case (no update, prototype persists unchanged).
- **Radius optimizer param group**: radii (`rho_c`) get their own `optim.param_group`
  with a separate (smaller, default 10x) learning rate and `weight_decay=0`,
  matching "own LR, no weight decay" in the mission text.
- **Malimg family "groups"** (variant families that would leak if split apart) are
  hardcoded in `configs/data/malimg_groups.yaml` from inspecting real Malimg folder
  names (Allaple.A/L, C2LOP.P/gen!g, Lolyda.AA1/AA2/AA3/AT, Swizzor.gen!C/gen!E,
  and the Yuner/Autorun/VB variants documented in the same config with a comment).
  This file is data, not code, so it can be corrected after inspecting the real
  Kaggle folder tomorrow without touching `splits.py`.
- **BIG 2015 intermediate cache resolution**: 256x256 uint8, per mission's explicit
  suggestion; final model input size is a separate resize step at load time so the
  cache is reusable across model presets.
- **Mock/synthetic data always writes `is_synthetic: true`** into `metrics.json` at
  the point the *dataset* is instantiated (propagated through to any run that
  touches it), not just at the top-level training script, so a baseline evaluated
  directly against a synthetic dataset also gets flagged correctly.
- **`aggregate` refuses synthetic runs** by filtering `metrics.json["is_synthetic"]
  is True` before any table/figure is built, and prints a loud count of how many
  runs were excluded, so a silent empty table doesn't get mistaken for "ran but
  found nothing".
- **Efficiency numbers (latency/throughput/memory) measured on whatever device the
  run actually used**, with `device` recorded; CPU numbers from today are honest
  CPU numbers, not GPU estimates — `experiments/queue.yaml` GPU-minute estimates are
  a *separate*, explicitly-labelled extrapolation (`estimated: true`) from CPU
  tiny_test throughput, never conflated with a measured run.
- **OSCR curve** follows the standard formulation (correct-classification rate on
  knowns vs. false-positive rate on unknowns, swept over the accept/reject
  threshold, AUC via trapezoidal integration) used across the open-set-recognition
  literature (Dhamija et al. 2018 and follow-ups); not the original 2000s OSCR
  paper's exact variant, since that one predates modern deep open-set benchmarks
  and the community has converged on this formulation.
- **OpenMax simplifications** (documented again here since they affect reported
  numbers): distance-to-mean-activation-vector uses cosine distance only (paper
  uses a Euclidean+cosine hybrid); no rank-based alpha-weighting of the Weibull
  revision scores. Both preserve the core recalibration mechanism the baseline
  exists to represent; see `baselines/openmax.py` docstring for the full algorithm
  citation.
- **Deep SVDD baseline shares the same `ViTBackbone` class as LRMC** (fair
  comparison, same preprocessing) rather than the original paper's bias-free,
  bounded-activation architecture that specifically blocks representation
  collapse. This is a real, documented risk — if a Deep SVDD run shows near-zero
  embedding variance, that is this simplification biting, not a bug to chase.
  Flagged again in HANDOFF.md's risk list.
- **t-SNE, not UMAP, for the embedding-projection figure** — scikit-learn's t-SNE
  is already a project dependency; adding `umap-learn` only for one figure isn't
  worth the extra install surface on a possibly-offline Kaggle session. Swap it in
  post-hoc if preferred; the figure code isolates the projection call to one line.
- **Radius-coverage figure draws 2D circles from a PCA projection** and says so in
  its own title — the true acceptance region is a cap on a high-dimensional
  hypersphere, not a disk in any 2D projection, so this figure is explicitly
  qualitative, never used as a source of quantitative claims.
- **Ablation epoch budget (15) is deliberately lower than the main run's (20)** —
  running all 23 ablations at the main run's budget would cost an estimated ~47
  GPU-hours (measured/extrapolated via `scripts/estimate_queue_budget.py`),
  exceeding Kaggle's 30 GPU-hours/week quota more than once over; ablations exist
  to show a qualitative direction of effect, not to match the main result's
  statistical power. See `experiments/queue.yaml`'s header comment for the numbers.
- **`run_baseline_evaluation` needs a `nearest_family(x)` method beyond the
  mission's literal fit/score/predict interface** — closed-set metrics and OSCR's
  `known_correct` need a family guess independent of the accept/reject threshold,
  which `predict(x)` alone can't provide once it returns `"UNKNOWN"`. Added as an
  extra method on every baseline (never a replacement for the required three), so
  the required interface still holds exactly as specified.
