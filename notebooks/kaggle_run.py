# %% [markdown]
# # LRMC Kaggle run notebook
#
# Cell-marker Python (`# %%`), openable as a notebook via Jupytext or by
# pasting cells into a Kaggle notebook directly. See HANDOFF.md for the full
# step-by-step. Summary: upload the bundle as a Dataset, attach Malimg/BIG2015
# as Datasets too, run every cell below in order, download the results zip.

# %% [markdown]
# ## Cell 1: detect environment and paths

# %%
import os
import sys
from pathlib import Path

IS_KAGGLE = Path("/kaggle").exists()
REPO_DIR = Path("/kaggle/working/lrmc") if IS_KAGGLE else Path.cwd()
print("IS_KAGGLE:", IS_KAGGLE)
print("REPO_DIR:", REPO_DIR)

if IS_KAGGLE:
    # The uploaded bundle lands under /kaggle/input/<dataset-slug>/. Adjust
    # BUNDLE_INPUT_DIR to match whatever slug Kaggle assigned on upload.
    BUNDLE_INPUT_DIR = Path("/kaggle/input/lrmc-kaggle-bundle")
    if not REPO_DIR.exists():
        import shutil

        shutil.copytree(BUNDLE_INPUT_DIR, REPO_DIR)
    os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR / "src"))
print("cwd:", Path.cwd())

# %% [markdown]
# ## Cell 2: install dependencies
#
# Kaggle's base image already has torch/torchvision/numpy/scipy/pandas/
# sklearn/matplotlib. We only need to add timm/py7zr, and we try pip first,
# falling back to a local wheelhouse (upload one as a second Dataset,
# `/kaggle/input/lrmc-wheelhouse/`, if the session has internet OFF).

# %%
import subprocess


def _pip_install(args: list[str]) -> bool:
    result = subprocess.run([sys.executable, "-m", "pip", "install", *args], capture_output=True, text=True)
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print(result.stderr[-2000:])
    return result.returncode == 0


packages = ["timm", "py7zr", "psutil"]
if not _pip_install(packages):
    print("Online pip install failed -- falling back to local wheelhouse.")
    wheelhouse = Path("/kaggle/input/lrmc-wheelhouse")
    if wheelhouse.exists():
        _pip_install(["--no-index", f"--find-links={wheelhouse}", *packages])
    else:
        print(
            "No wheelhouse found at",
            wheelhouse,
            "-- if this session truly has no internet, upload wheels there first.",
        )

# %% [markdown]
# ## Cell 3: detect dataset paths and point configs at them
#
# Edit MALIMG_ROOT / BIG2015_ROOT to match whatever your attached Kaggle
# Datasets are actually named (visible under /kaggle/input/).

# %%
import glob

candidates_malimg = glob.glob("/kaggle/input/*malimg*")
candidates_big2015 = glob.glob("/kaggle/input/*malware-classification*") + glob.glob(
    "/kaggle/input/*big2015*"
)
print("Malimg candidates:", candidates_malimg)
print("BIG2015 candidates:", candidates_big2015)

MALIMG_ROOT = candidates_malimg[0] if candidates_malimg else "/kaggle/input/malimg-dataset"
BIG2015_ROOT = candidates_big2015[0] if candidates_big2015 else "/kaggle/input/microsoft-malware-classification-challenge"
print("Using MALIMG_ROOT =", MALIMG_ROOT)
print("Using BIG2015_ROOT =", BIG2015_ROOT)

# %% [markdown]
# ## Cell 4: patch base configs with the real paths + device

# %%
from lrmc.config import load_config, save_config

for cfg_path, root, cache in [
    ("configs/base_malimg.yaml", MALIMG_ROOT, "/kaggle/working/cache_malimg"),
    ("configs/base_big2015.yaml", BIG2015_ROOT, "/kaggle/working/cache_big2015"),
]:
    cfg = load_config(cfg_path)
    cfg.data.root = root
    cfg.data.cache_dir = cache
    save_config(cfg, cfg_path)
    print(f"patched {cfg_path}: data.root={root}")

# %% [markdown]
# ## Cell 5: sanity-check the dataset loaders before spending GPU time

# %%
from lrmc.data.malimg import MalimgLoader

malimg_loader = MalimgLoader(MALIMG_ROOT)
print("Malimg report:", malimg_loader.report)
assert not malimg_loader.report.mismatches, "STOP: Malimg layout does not match expectations -- fix data.root."

# %% [markdown]
# ## Cell 6: run the queue
#
# `run-queue` executes Tier A -> B -> C in priority order, skips any job
# whose runs/<name>/metrics.json already exists (so re-running this cell
# after a session gets interrupted just resumes), and stops gracefully when
# the remaining session budget can't cover the next job.

# %%
from lrmc.orchestration.queue import QueueRunner

runner = QueueRunner("experiments/queue.yaml", session_budget_minutes=660)  # ~11h, under the 12h cap
results = runner.run()
for r in results:
    print(f"[{r.tier}] {r.name}: {r.status} (est={r.estimated_gpu_minutes}, actual={r.elapsed_minutes})")

# %% [markdown]
# ## Cell 7: aggregate into paper_artifacts/ and zip the results

# %%
from lrmc.aggregate.collect import collect_runs
from lrmc.aggregate.facts import write_paper_facts
from lrmc.aggregate.figures import generate_all_figures
from lrmc.aggregate.summary import write_results_summary
from lrmc.aggregate.tables import generate_all_tables

runs = collect_runs("runs")
generate_all_tables(runs, "paper_artifacts/tables")
generate_all_figures(runs, "paper_artifacts/figures")
write_paper_facts(runs, "paper_artifacts/paper_facts.json")
write_results_summary(runs, "paper_artifacts/results_summary.md")

# %%
import shutil

shutil.make_archive("/kaggle/working/lrmc_results", "zip", root_dir=".", base_dir="runs")
shutil.make_archive("/kaggle/working/lrmc_paper_artifacts", "zip", root_dir=".", base_dir="paper_artifacts")
print("Download /kaggle/working/lrmc_results.zip and lrmc_paper_artifacts.zip from the Output tab.")
