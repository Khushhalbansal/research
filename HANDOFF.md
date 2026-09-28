# HANDOFF: running this on Kaggle

Everything in this repo was written and tested on CPU with mock data today
(2026-09-28). 121 tests pass; a full rehearsal (`lrmc run-queue
experiments/queue_rehearsal.yaml` + `lrmc aggregate`) ran end-to-end with zero
code changes needed. This document is the exact sequence for tomorrow's real
GPU session, plus what to check first if a result looks wrong.

## 1. Before you start: what to attach

On kaggle.com, create a new Notebook, then under "Add Input":

1. **The code bundle.** Run `bash scripts/make_kaggle_bundle.sh` locally (or
   `python scripts/make_kaggle_bundle.py`) to produce `lrmc_kaggle_bundle.zip`.
   Upload it as a new Kaggle **Dataset** (Datasets -> New Dataset -> Upload).
   Note the slug Kaggle assigns (visible in the dataset URL); the notebook
   guesses `/kaggle/input/lrmc-kaggle-bundle` but auto-detection in Cell 1 of
   `notebooks/kaggle_run.py` will need adjusting if the slug differs.
2. **Malimg.** Attach a Malimg dataset (search Kaggle Datasets for "malimg" --
   several public mirrors exist). The loader auto-detects the folder layout,
   but verify with the report in Cell 5 of the notebook before spending GPU
   time.
3. **BIG 2015 (Microsoft Malware Classification Challenge).** Attach it if
   you want Tier C; the competition data may need accepting competition rules
   first. Files may be `.7z` archives -- the loader streams-extracts these
   with a disk-budget check (`Big2015Loader`), verified on mock `.bytes`
   files today but never against the real archives (see risk list below).
4. **(Optional) A wheelhouse dataset** named `lrmc-wheelhouse` containing
   `.whl` files for `timm`, `py7zr`, `psutil` if the session has internet
   OFF. Build one locally with `pip download timm py7zr psutil -d wheelhouse/`
   and upload that folder as a Dataset.
5. **GPU + internet.** In the notebook's right sidebar: Accelerator = GPU
   (T4 x2), and leave internet ON unless you have a specific reason not to --
   the pretrained-weight fallback (see risk list) degrades gracefully either
   way, but ON is simpler.

## 2. Running it

Open `notebooks/kaggle_run.py` in Kaggle (paste its cells into a new
notebook, or convert with `jupytext --to notebook notebooks/kaggle_run.py` if
you have jupytext locally) and run cells 1-7 in order:

1. Detects `/kaggle/input`, copies the bundle to `/kaggle/working/lrmc`.
2. Installs `timm`/`py7zr`/`psutil` (pip, falling back to the wheelhouse).
3. Auto-detects Malimg/BIG2015 dataset paths under `/kaggle/input`.
4. Patches `configs/base_malimg.yaml` / `configs/base_big2015.yaml` with the
   real paths.
5. **Sanity-checks the Malimg loader report before spending any GPU time** --
   this cell has a hard `assert` that the layout matches (25 families, 9339
   images). If it fails, STOP and fix `data.root` / inspect the folder
   structure before continuing; do not comment out the assert.
6. Runs the queue: `QueueRunner("experiments/queue.yaml", session_budget_minutes=660)`.
7. Aggregates into `paper_artifacts/` and zips `runs/` + `paper_artifacts/`
   for download from the notebook's Output tab.

Equivalently, from a terminal inside the Kaggle notebook (Kaggle notebooks
support a terminal via the "..." menu, or use `!` shell cells):

```bash
cd /kaggle/working/lrmc
pip install -e .
lrmc run-queue experiments/queue.yaml
lrmc aggregate
```

If the session is interrupted (12-hour limit, or you just close the tab),
**re-running the exact same `lrmc run-queue experiments/queue.yaml` command is
always safe** -- completed jobs are skipped (`runs/<name>/metrics.json`
already exists), and training itself resumes mid-run from the last
checkpoint if a `train_lrmc` job was interrupted partway (proven by
`tests/integration/test_resume_equivalence.py`).

## 3. Expected runtime per tier

These are **honest extrapolations**, not measurements (see
`src/lrmc/orchestration/budget.py` and `scripts/estimate_queue_budget.py`):
measured CPU tiny_test forward+backward time, scaled by the measured
approx-FLOPs ratio to the real backbone, divided by an assumed 10x CPU->T4
speedup (conservative; DataParallel across the 2 T4s is NOT assumed to double
throughput). Re-run `python scripts/estimate_queue_budget.py` after landing
on Kaggle to get numbers grounded in the actual GPU if you want to re-tune
`session_budget_minutes`.

| Tier | Contents | Estimated | Fits in |
|---|---|---|---|
| A | Malimg main LRMC run + 4 trained baselines (softmax/energy/OpenMax/DeepSVDD) + 5 embedding baselines (prototype-cosine, fixed-radius, Mahalanobis, kNN, OCSVM) | ~640 min (~10.7h) | one 12h session (tight) |
| B | 23 ablation configs, 15 epochs each | ~2840 min (~47h) | 2 weekly 30h quotas |
| C | BIG2015 main + 2 extras | ~245 min (~4h) | one session, after A |

`session_budget_minutes: 660` in `experiments/queue.yaml` leaves an ~1h
safety margin under the 12h cap for Tier A. Tier B is *designed* to span
multiple sessions/weeks -- the runner stops (doesn't skip ahead) once the
remaining budget can't cover the next job's estimate, so priority order is
preserved across however many sessions it takes.

## 4. What to check first if a result looks off

In roughly the order to check them:

1. **AUROC suspiciously close to 1.0 on the very first run.** The #1 signal of
   train/test leakage. Check: (a) the split protocol actually used
   (`runs/<name>/fold.json` -> `protocol`, `known_families`,
   `unknown_families` -- confirm the unknown families are genuinely absent
   from `train`/`val_known`); (b) for Malimg specifically, whether a
   duplicate-heavy family straddled the split (the loader's
   `n_duplicate_files`/`n_duplicate_hashes` report should be sane, and
   `data/splits.py`'s hash-group-level split is supposed to prevent this --
   if you suspect it didn't, run `assert_fold_disjoint` manually against the
   fold and the loader's records); (c) `split.group_aware` -- if `false`
   (naive holdout) on Malimg, a held-out "unknown" family might have a
   near-duplicate sibling still in `known_families` (e.g. Allaple.A held out
   but Allaple.L still known), which is easy known-vs-unknown and inflates
   AUROC without being a bug -- compare against the `group_aware_holdout`
   variant of the same experiment to see the gap.
2. **Radii collapse (all `r_c` -> ~0 or all -> very large).** Look at
   `paper_artifacts/figures/radii_over_epochs_<run>.png`. Near-zero radii
   with `l_in` loss still low means the encoder collapsed embeddings into a
   tiny region (check `training_curves_<run>.png` -- `l_in`/`l_out` both
   near zero simultaneously is a red flag, not a good sign). Very large radii
   mean `gamma` (the tightness weight) is too small relative to `alpha` --
   per docs/LRMC.md section 5, `r_c` should sit near the
   `(1 - gamma/alpha)`-quantile of in-class distances; if it's way off that,
   something is wrong with the loss weighting for that config, not
   necessarily the implementation (compare against
   `ablation_gamma_*` runs and `plot_gamma_vs_frr` in the aggregated
   figures, which tests this exact relationship on real results).
3. **Prototype drift / instability.** If `radii_over_epochs` looks fine but
   closed-set accuracy is poor, check the EMA momentum (`prototypes.momentum`,
   default 0.9) -- a momentum too low makes prototypes chase noisy per-batch
   means; try `prototypes_batch_mean.yaml` as a comparison (no momentum at
   all) to see whether that's the driver.
4. **A baseline beats LRMC.** Not a bug by default -- report it (see
   `results_summary.md`'s explicit negative-result callout). But do check:
   was the baseline's own network (softmax/OpenMax/DeepSVDD) actually
   converged? These get the SAME epoch budget as LRMC in
   `experiments/queue.yaml` by default; if LRMC needs more epochs to
   converge than a simple softmax classifier does, that's a legitimate
   training-budget confound worth calling out explicitly, not hiding.
5. **`pretrained_source` in `metrics.json` is `"random_init"` when you
   expected `"timm_pretrained"`.** Means the ImageNet-pretrained ViT weights
   failed to download (internet off, or a Kaggle-side hiccup) and training
   silently fell back to random init -- silently in the sense that training
   still runs, but every number downstream is now a from-scratch-ViT result,
   not a pretrained one. This is logged loudly (`logger.warning`) at train
   time; check the notebook's cell 6 output, not just the final metrics.
6. **A `baseline_embedding` job fails with "dependency job ... has not
   produced a checkpoint".** Means its paired `train_lrmc` job (referenced by
   `depends_on` in `experiments/queue.yaml`) hasn't completed yet -- check
   Tier A's job order; `run-queue` executes jobs in file order within a
   tier, so this should only happen if an earlier job failed. Check
   `runs/queue_state.json` for the `error` field on the dependency job.

## 5. Known risks and untested-without-GPU-or-real-data items

Everything below was written to be correct by construction and unit-tested
against synthetic analogues, but genuinely **could not be verified today**:

- **Real Malimg / BIG2015 folder layouts.** `MalimgLoader`'s auto-detection
  and `Big2015Loader`'s `.7z` streaming + `trainLabels.csv` join are tested
  against mock data shaped like the real thing, never the real archives.
  Cell 5's pre-flight assert exists specifically to catch a layout mismatch
  before wasting GPU time.
- **ImageNet-pretrained ViT download on Kaggle's actual network policy.**
  Never tested against Kaggle's specific proxy/firewall; the three-tier
  fallback (timm download -> local weights -> random init) is unit-tested
  by forcing failures, but the *real* failure mode on Kaggle is unknown
  until you run it.
- **Multi-GPU DataParallel path.** `Trainer` wraps the feature-extraction
  network in `nn.DataParallel` when `torch.cuda.device_count() > 1`; this
  branch has never executed (no GPU available today). Watch the first Tier A
  job closely for anything unusual in per-GPU memory or throughput.
- **Actual GPU-minute costs.** Every number in `experiments/queue.yaml` is
  an extrapolation from CPU timing (see docs/DECISIONS.md); the *real* T4
  throughput could differ meaningfully in either direction. Re-run
  `scripts/estimate_queue_budget.py`'s logic against a real GPU early in
  the session if you want a corrected `session_budget_minutes`.
- **7z extraction disk usage on the real BIG2015 archive.** The ~20GB soft
  budget check (`Big2015Loader._extract_archives`) has only been exercised
  against small synthetic files; the real archive's actual extracted size
  is unverified.
- **Deep SVDD representation collapse** (see docs/DECISIONS.md) -- this
  baseline's simplified architecture (shares `ViTBackbone`, not the
  bias-free/bounded-activation design the original paper uses to block
  collapse) is a real risk on real data, not just theoretical.
- **OpenMax Weibull fits with too few correctly-classified training samples
  per class.** Falls back to "no revision" for that class (fail-soft,
  documented in `baselines/openmax.py`), which is safe but could silently
  weaken the OpenMax baseline for rare Malimg families -- check
  `weibull_params` isn't `None` for most classes if OpenMax looks weak.

## 6. Top 5 risks to scientific validity (read before trusting any number)

1. **Open-set leakage via near-duplicate/variant families**, especially under
   naive (non-group-aware) holdout on Malimg. This is exactly why
   `group_aware_holdout` exists and why `experiments/queue.yaml` includes
   both `split_naive_holdout` (ablation) and the group-aware main run --
   report the gap between them, don't just report the better number.
2. **Undertrained baselines making LRMC look artificially strong.** All
   baselines share LRMC's epoch budget by default (`params: { epochs: 15 }`
   in `experiments/queue.yaml`); if any baseline's training loss (visible in
   its own logged history, though baselines don't currently persist a
   training-curve figure the way LRMC does) hasn't plateaued, extend its
   epoch budget and re-run before trusting a comparison against it.
3. **Radius/gamma miscalibration reading as a model failure.** Section 5 of
   docs/LRMC.md predicts a specific, checkable relationship
   (`r_c ~ (1-gamma/alpha)`-quantile); if real results violate it, that is
   informative (possibly `beta`'s L_out term dominating, as the derivation's
   caveat notes) rather than necessarily a code bug -- check
   `plot_gamma_vs_frr` before assuming something is broken.
4. **Pretrained-weight fallback changing silently.** As above (#5 in section
   4) -- a `random_init` fallback on what you assumed was a pretrained run
   invalidates any claim that compares against the literature's pretrained-ViT
   numbers. Always check `pretrained_source` in `metrics.json`.
5. **Small effective known-class counts under high-openness ablations**
   (`n_unknown_5`, `n_unknown_8`) mean both smaller training sets AND smaller
   test-known sets -- metric variance goes up as openness increases, and a
   single-seed run at high openness is not statistically trustworthy on its
   own. `experiments/queue.yaml` currently runs one seed per ablation; if you
   have spare budget, prioritize adding seeds to the openness sweep over
   adding new ablation dimensions.

## 7. After the run

```bash
lrmc aggregate   # rebuilds paper_artifacts/ from whatever is in runs/
```

Check `paper_artifacts/results_summary.md` first -- it states plainly whether
LRMC beat or lost to the best baseline, and calls out the fixed-vs-learnable
radius delta directly. Then pull numbers for the paper from
`paper_artifacts/tables/*.tex` (`\input` these directly) and
`paper_artifacts/paper_facts.json` for the Experimental Setup section.
`paper_artifacts/figures/gamma_vs_frr.png` is the direct empirical test of
the docs/LRMC.md section 5 claim on real data -- worth a look even if it's
not going in the paper, as a sanity check that the implementation is behaving
as derived.
