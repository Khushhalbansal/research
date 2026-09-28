# HANDOFF: running this on the college GPU workstation

Everything in this repo was written and tested on CPU with mock data (2026),
then adapted for a physical NVIDIA GPU workstation reached over AnyDesk whose
OS, GPU model/VRAM, disk space, internet access, and admin rights are all
**unknown until you actually connect**. 165 tests pass; a full rehearsal
(`lrmc run-queue experiments/queue_rehearsal.yaml` + `lrmc aggregate`) runs
end-to-end on mock data with zero code changes needed. This document is the
exact sequence for that machine, plus what to check first if a result looks
wrong. Kaggle remains usable as an optional profile (`notebooks/kaggle_run.py`,
`--override data.root=/kaggle/...`) but is no longer the assumed target --
skip section 12 unless you're specifically going back to Kaggle.

## 0. First-connection checks (before installing anything)

Once AnyDesk connects, before running any command, find out:

```
# Windows (PowerShell)
$PSVersionTable.PSVersion          # PowerShell version
Get-ComputerInfo | Select WindowsProductName, OsArchitecture
nvidia-smi                         # GPU model, VRAM, driver/CUDA version (if present)
python --version                   # or python3 --version, or py --version
Get-PSDrive -PSProvider FileSystem # free disk per drive
whoami /priv | findstr /i "admin"  # rough admin-rights check
```

```bash
# Linux
uname -a
nvidia-smi
python3 --version
df -h
whoami; groups   # look for sudo/wheel/admin group membership
```

Write these down -- they determine which flags you'll pass to
`scripts/setup_env.py` (`--no-venv` if no admin, `--offline` if no internet)
and what `device: auto` will actually resolve to.

## 1. Security & hygiene (read once, matters for the whole session)

- **This codebase never executes a sample.** `FilePreprocessor.load_raw_executable`
  and the BIG2015 `.bytes` loader only ever call `open(...).read()`; the
  optional PE-header check (`pe_header_sanity_check`) is a passive byte
  comparison, not a parse-to-execute. `scan` and `verify-data` are read-only.
- **Antivirus will very likely flag the BIG2015 `.bytes`/raw-executable
  samples** -- they're real malware bytes (BIG2015) or synthetic
  test executables. Put the dataset under a **dedicated folder** (e.g.
  `data/big2015/`, `data/malimg/`) and ask the college IT admin for an AV
  **exclusion on that folder specifically** -- not a blanket AV disable.
  Do this before `fetch_data.py`/copying data in, or the AV may quarantine
  files mid-download/mid-extraction.
- **Malimg is images, not executables.** The Malimg dataset is pre-rendered
  grayscale PNGs -- normal image files, nothing for an AV to flag, no
  exclusion needed for that folder.
- Never disable antivirus workstation-wide for this project; the narrow
  folder exclusion is both sufficient and the more defensible ask to IT.

## 2. Setup

```bash
git clone <this repo>            # or copy the Kaggle-bundle zip and unzip
cd research
python scripts/setup_env.py                    # normal case: admin + internet
python scripts/setup_env.py --no-venv           # no admin rights
python scripts/setup_env.py --offline           # no internet (needs ./wheelhouse, see below)
python scripts/setup_env.py --proxy http://host:port   # behind a proxy
```

If the workstation has no/restricted internet: at home, on a machine with the
**same OS and Python major.minor version**, run
`pip download -r requirements.txt -d wheelhouse` (and `requirements-dev.txt`
too if you want pytest/ruff/black), copy the `wheelhouse/` folder over
(USB drive or AnyDesk file transfer), then use `--offline` or
`--wheelhouse <path>`.

The script ends by printing GPU name/VRAM/CUDA version/free disk/Python
version -- confirm these match what you saw in step 0 before continuing.

```bash
python -m pytest -q       # confirm the suite still passes on THIS machine
```

## 3. Get the data

**Automatic (Kaggle API):**
```bash
pip install kaggle   # or: pip install -e .[kaggle-fetch]
# place kaggle.json at ~/.kaggle/kaggle.json (Linux) or
# %USERPROFILE%\.kaggle\kaggle.json (Windows) -- from Kaggle: Settings -> API
python scripts/fetch_data.py --dataset both
```
This pre-checks free disk space, skips the download if a matching-size zip
already exists (idempotent -- see the script's docstring for exactly what
"resumable" means here), extracts, and runs `verify-data` automatically.

**Manual (no Kaggle API, or no internet at all):** copy the data off a USB
drive / network share into `data/malimg/` and `data/big2015/` yourself
(into the AV-excluded folder from section 1), then:
```bash
python -m lrmc verify-data --dataset malimg  --root data/malimg
python -m lrmc verify-data --dataset big2015 --root data/big2015
```
Both reproduce the layout/count/duplicate report (25 families / 9339 images
for Malimg; ~10.8k labeled / 9 families for BIG2015) and **exit 1 on any
mismatch** -- fix `data.root`/the folder layout before spending GPU time,
don't proceed past a mismatch.

## 4. Benchmark (measure the real GPU, don't guess)

```bash
python -m lrmc benchmark --device auto --seeds 5
```
This measures real throughput and max batch size on the actual GPU (or CPU,
if that's genuinely all there is), rewrites every `estimated_gpu_minutes` in
`experiments/queue.yaml` as `estimated: false`, re-prioritizes each tier, and
-- since a dedicated workstation has no Kaggle-style weekly quota --
**expands Tier A and Tier B with 5-seed repeats** (generated configs land in
`configs/generated/`). The original queue is backed up once to
`experiments/queue.yaml.pre_benchmark_backup`. Re-run this if you ever change
the backbone/epoch counts in `configs/base_*.yaml`.

If Tier B's expanded size (23 ablations x 5 seeds = 115 jobs) is more than
you want to commit to, edit `experiments/queue.yaml` after benchmarking to
trim it, or re-run with `--seeds 3` for the mission's minimum instead of the
target 5.

## 5. Smoke run before the real thing

```bash
python -m lrmc train configs/rehearsal_mock_malimg.yaml
python -m lrmc evaluate configs/rehearsal_mock_malimg.yaml
```
Confirms the real GPU path (CUDA, DataParallel if >1 GPU, AMP autocast)
works end-to-end on this specific machine before committing to a multi-hour
Tier A run. Should take well under a minute even on a modest GPU.

## 6. Launch Tier A (survives an AnyDesk disconnect)

```bash
# Linux
scripts/run_detached.sh experiments/queue.yaml

# Windows (PowerShell)
powershell -File scripts\run_detached.ps1 -Queue experiments/queue.yaml
```
Both launch `lrmc run-queue` detached from the current session, logging to
`logs/run_queue_<timestamp>.log`. A PID lock (`runs/.queue.lock`) stops a
second run-queue from starting by accident after you reconnect and forget
one is already going. If a job is interrupted mid-training, re-running the
exact same command is always safe: completed jobs are skipped
(`runs/<name>/metrics.json` exists), and an interrupted `train_lrmc` job
resumes from its last checkpoint automatically (proven by
`tests/integration/test_resume_equivalence.py`). A CUDA OOM (e.g. someone
else grabs the GPU) shrinks the batch size, grows gradient accumulation to
compensate, and retries -- logged, not fatal.

## 7. Check on it after reconnecting

```bash
python -m lrmc status
```
Prints: current job (tier/type), epoch progress + ETA, last checkpoint age,
overall queue progress, per-GPU utilization/free VRAM, free disk. No log
tailing required. If you do want the raw log:
```bash
tail -f logs/run_queue_<timestamp>.log          # Linux
Get-Content -Wait -Tail 30 logs\run_queue_<ts>.log   # Windows
```

## 8. What to check first if a result looks wrong

Same triage list as before, still applicable:

1. **AUROC suspiciously close to 1.0 on the very first run.** Check
   `runs/<name>/fold.json` for the actual `protocol`/`known_families`/
   `unknown_families` used, and whether `split.group_aware` was `true`
   (naive holdout can let a near-duplicate variant family leak).
2. **Radii collapse (all `r_c` -> ~0 or -> very large).** Look at
   `paper_artifacts/figures/radii_over_epochs_<run>.png`; cross-check against
   docs/LRMC.md section 5's `(1-gamma/alpha)`-quantile prediction via
   `plot_gamma_vs_frr`.
3. **Prototype drift / instability.** Compare against the
   `ablation_prototypes_batch_mean` run (no EMA memory) to isolate whether
   `prototypes.momentum` is the driver.
4. **A baseline beats LRMC.** Report it -- check `results_summary.md`'s
   explicit negative-result callout -- but first confirm the baseline's own
   network actually converged at its epoch budget.
5. **`pretrained_source` in `metrics.json` is `"random_init"` when you
   expected `"timm_pretrained"`.** The ImageNet-weight download failed
   (offline session, or a hiccup) and training silently fell back to
   scratch -- logged loudly at train time, easy to miss in a detached run's
   log. Always check this field before trusting a pretrained-vs-literature
   comparison.
6. **A `baseline_embedding` job fails with "dependency job ... has not
   produced a checkpoint".** Its paired `train_lrmc` job hasn't finished yet
   (or failed) -- check `runs/queue_state.json` for that job's `error` field.
7. **A job fails with a CUDA error mentioning another process / driver
   mismatch.** The GPU may be shared with someone else's job. `lrmc status`
   shows per-GPU utilization; `resolve_device("auto")` already picks the
   least-loaded GPU when more than one is present, but on a single-GPU
   workstation there's nowhere else to go -- coordinate with whoever else
   might be using it.

## 9. Copying results home and building the paper artifacts

From the workstation:
```bash
python -m lrmc aggregate            # rebuilds paper_artifacts/ from runs/
zip -r results.zip runs paper_artifacts    # Linux
Compress-Archive -Path runs,paper_artifacts -DestinationPath results.zip  # Windows
```
Then get `results.zip` home however is easiest: AnyDesk's file-transfer
panel, a cloud-storage upload (Drive/Dropbox/OneDrive -- upload from the
workstation's browser or CLI, whichever is already authenticated there), or
a USB drive if you're physically at the machine.

At home:
```bash
unzip results.zip -d .
python -m lrmc aggregate      # regenerates paper_artifacts/ if you add more runs later
```
Check `paper_artifacts/results_summary.md` first -- it states plainly
whether LRMC beat or lost to the best baseline and calls out the
fixed-vs-learnable radius delta directly. Pull numbers for the paper from
`paper_artifacts/tables/*.tex` (`\input` these directly) and
`paper_artifacts/paper_facts.json` for the Experimental Setup section.

## 10. Known risks and untested-without-real-GPU items

Everything below was written to be correct by construction and unit-tested
against synthetic/CPU analogues, but genuinely **could not be verified
without the actual target machine**:

- **Real Malimg / BIG2015 folder layouts** on THIS machine's copy of the
  data -- `verify-data`'s pre-flight check exists specifically to catch a
  mismatch before spending GPU time; don't skip it.
- **ImageNet-pretrained ViT download on this workstation's actual network
  policy** (proxy, firewall, IT-imposed allowlist). The three-tier fallback
  (timm download -> local weights -> random init) is unit-tested by forcing
  failures, but the *real* failure mode here is unknown until you run it.
- **Multi-GPU `DataParallel`.** `Trainer` wraps the feature-extraction
  network in `nn.DataParallel` only when `torch.cuda.device_count() > 1`;
  this branch has never executed (no multi-GPU machine available while
  building this). If the workstation has 2+ GPUs, watch the first job
  closely for anything unusual in per-GPU memory or throughput scaling.
- **CUDA OOM backoff** (`Trainer._handle_cuda_oom`) is unit-tested by
  monkeypatching a `RuntimeError("CUDA out of memory")`, never against a
  real allocator failure -- the logic (halve batch, double grad
  accumulation, rebuild the DataLoader, retry) is straightforward, but a
  real OOM's exact failure point mid-step is untested.
- **`nvidia-smi`-based GPU selection/status** (`utils/hardware.py`)
  depends on `nvidia-smi` being on PATH with the expected CSV output format
  -- verified against this tool's actual behavior on a real NVIDIA driver
  install only once you run it here; falls back to `torch.cuda.mem_get_info`
  if `nvidia-smi` isn't found, and to "unavailable" if neither works.
- **`scripts/setup_env.py`'s `--no-venv`/`--proxy`/`--wheelhouse` paths**
  are individually straightforward but have never all been exercised
  together against a real locked-down Windows image.
- **7z extraction disk usage on the real BIG2015 archive** -- the ~20GB soft
  budget check (`Big2015Loader._extract_archives`) has only been exercised
  against small synthetic files.
- **Deep SVDD representation collapse** (see docs/DECISIONS.md) -- this
  baseline's simplified architecture is a real risk on real data, not just
  theoretical.
- **`scripts/fetch_data.py`'s Kaggle slugs** (`DEFAULT_MALIMG_SLUG`,
  `DEFAULT_BIG2015_SLUG`) point at commonly-used public mirrors as of
  writing; Kaggle dataset slugs occasionally change owners or get taken
  down -- if the download 404s, search Kaggle for a current Malimg mirror
  and pass `--malimg-slug`.

## 11. Top 5 risks to scientific validity (read before trusting any number)

1. **Open-set leakage via near-duplicate/variant families**, especially
   under naive (non-group-aware) holdout on Malimg -- report the gap
   between `split_naive_holdout` and the group-aware main run, don't just
   report the better number.
2. **Undertrained baselines making LRMC look artificially strong.** All
   baselines share LRMC's epoch budget by default; if a baseline's training
   loss hasn't plateaued, extend its epoch budget before trusting a
   comparison against it.
3. **Radius/gamma miscalibration reading as a model failure.** Check
   `plot_gamma_vs_frr` against docs/LRMC.md section 5's prediction before
   assuming something is broken.
4. **Pretrained-weight fallback changing silently** -- always check
   `pretrained_source` in `metrics.json`.
5. **Small effective known-class counts under high-openness ablations**
   (`n_unknown_5`, `n_unknown_8`) -- with the 5-seed expansion from section 4
   this is now better-powered than the original single-seed Kaggle plan,
   but still worth a wider CI on the headline numbers.

## 12. Kaggle (optional, legacy profile)

If you ever want to go back to Kaggle instead of (or in addition to) the
workstation: `notebooks/kaggle_run.py` still works as originally written,
and any config can be pointed at Kaggle paths with
`--override data.root=/kaggle/input/... --override data.cache_dir=/kaggle/working/cache`.
`experiments/queue.yaml`'s job list is dataset/protocol-identical either
way -- only the paths and the GPU-minute estimates (re-measure with
`lrmc benchmark` on whichever machine you're actually using) differ.
