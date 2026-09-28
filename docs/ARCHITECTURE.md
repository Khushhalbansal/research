# Architecture

This document maps the two diagrams in `/paper` (`methodology.jpeg` = training,
`inference diagram.jpeg` = inference) to the code layout under `src/lrmc/`. Class and
module names mirror the diagram box names exactly so the paper's figures and the code
never drift apart.

## Diagram -> code map

### Training (methodology.jpeg)

`Data Ingestion and Preprocessing Module` -> `Feature Extraction Network` ->
`LRMC Training Module` (loss feeds gradients back into the backbone/head).

| Diagram box | Module | Class |
|---|---|---|
| File Preprocessor | `lrmc.data.preprocessor` | `FilePreprocessor` |
| Image Generator | `lrmc.data.image_generator` | `ImageGenerator` |
| Vision Transformer Backbone | `lrmc.models.backbone` | `ViTBackbone` |
| Projection Head | `lrmc.models.projection_head` | `ProjectionHead` |
| Embedding Normalizer | `lrmc.models.normalizer` | `EmbeddingNormalizer` |
| Class Prototype Estimator | `lrmc.models.prototypes` | `ClassPrototypeEstimator` |
| LRMC Loss Calculator | `lrmc.losses.lrmc_loss` | `LRMCLossCalculator` |
| Learnable Radii Parameters | `lrmc.models.radii` | `LearnableRadiiParameters` |

`lrmc.models.network.FeatureExtractionNetwork` bundles Backbone + Projection Head +
Embedding Normalizer, since the diagram treats them as one box group used identically
in both training and (frozen) inference.

### Inference (inference diagram.jpeg)

`Data Ingestion and Preprocessing Module` -> `Fixed Feature Extraction Network` ->
`Radius Based Decision Engine` -> `Alert Generator`.

| Diagram box | Module | Class |
|---|---|---|
| Stored Class Prototypes | `lrmc.models.prototypes` | `ClassPrototypeEstimator` (frozen, loaded from checkpoint) |
| Stored Learnable Radii | `lrmc.models.radii` | `LearnableRadiiParameters` (frozen) |
| Distance Calculator | `lrmc.engine.infer` | `DistanceCalculator` |
| Zero Day Classifier | `lrmc.engine.infer` | `ZeroDayClassifier` |
| Alert Generator | `lrmc.engine.alert` | `AlertGenerator` |

## Package layout

```
src/lrmc/
  config.py              # dataclass configs, YAML loading, config hashing
  utils/
    seed.py, env_info.py  # seeding, package/env-version capture, git commit
    env_detect.py           # "kaggle"/"colab"/"local" detection (informational only)
    hardware.py              # resolve_device (incl. least-loaded-GPU pick), resolve_num_workers,
                              #   gpu_status_list, free_disk_bytes, system_summary
  data/
    preprocessor.py       # FilePreprocessor (BIG .bytes parser, raw-exe scan path)
    image_generator.py    # ImageGenerator (Nataraj width table, resize, two-view aug)
    augmentations.py      # byte-image-safe augmentations (no h-flip)
    malimg.py              # Malimg loader: auto-detect layout, dedup, imbalance report
    big2015.py             # BIG 2015 loader: 7z streaming, memmap/npz cache, manifest
    mock.py                 # synthetic Malimg-style + BIG-style generators
    splits.py               # leave-one-family-out / random-k-unknown / group-aware holdout
    datasets.py              # torch Dataset wrappers, class-balanced sampler
  models/
    backbone.py            # ViTBackbone (timm presets incl. tiny_test, resnet18)
    projection_head.py     # ProjectionHead (2-layer MLP)
    normalizer.py          # EmbeddingNormalizer
    prototypes.py          # ClassPrototypeEstimator (EMA / batch-mean, lazy init)
    radii.py                # LearnableRadiiParameters (softplus reparam)
    network.py               # FeatureExtractionNetwork (backbone+head+normalizer)
  losses/
    supcon.py                # SupConLoss (Khosla et al.)
    lrmc_loss.py              # LRMCLossCalculator (supcon + alpha*L_in + beta*L_out + gamma*L_rad)
  engine/
    train.py                  # Trainer: loop, checkpoint, resume, grad accumulation
    checkpoint.py             # save/load model+optim+prototypes+radii+RNG+epoch
    infer.py                   # DistanceCalculator, ZeroDayClassifier, frozen inference
    scan.py                     # raw-executable scan path (read-only, size-limited)
    alert.py                     # AlertGenerator -> JSONL
  eval/
    metrics.py                  # closed-set + open-set metrics, OSCR, bootstrap CI
    efficiency.py                # params, approx FLOPs, latency/throughput/memory
    calibration.py                # post-hoc radius calibration ablation
  baselines/
    base.py                      # BaselineDetector: fit/score/predict interface
    softmax_msp.py, odin_energy.py, openmax.py, mahalanobis.py, knn.py,
    ocsvm.py, deep_svdd.py, fixed_radius_prototype.py, prototype_cosine.py
  orchestration/
    queue.py                      # priority job runner over experiments/queue.yaml (PID-locked)
    budget.py                      # CPU-throughput -> GPU-minute estimate (pre-GPU, honest extrapolation)
    benchmark.py                    # real GPU throughput/max-batch measurement, queue rewrite, multi-seed expansion
    lock.py                          # QueueLock: PID-file lock, stale-lock reclaim
    status.py                         # status.json (per-epoch) / current_job.json, read by `lrmc status`
  aggregate/
    collect.py, tables.py, figures.py, facts.py, summary.py
  cli/
    main.py    # train / evaluate / infer / scan / run-queue / verify-data / status / benchmark / aggregate
```

## Data flow contracts

* `FilePreprocessor.load(path) -> np.ndarray[uint8]` — raw byte stream (BIG `.bytes`
  with `??` handling) or pre-rendered image bytes (Malimg), never executes input.
* `ImageGenerator.to_image(byte_array, size) -> np.ndarray[uint8, size, size]` — Nataraj
  width table for byte streams; passthrough+resize for existing images.
* `ImageGenerator.two_views(image) -> (view1, view2)` — augmentation pair for SupCon.
* `FeatureExtractionNetwork.forward(image) -> z (L2-normalized, dim=embed_dim)`.
* `ClassPrototypeEstimator.update(z, y)` (train) / `.prototypes` (frozen dict at
  inference, `class_id -> unit vector`).
* `LearnableRadiiParameters.radii -> {class_id: r_c}` via `softplus(rho_c)`.
* `LRMCLossCalculator(z, y, prototypes, radii) -> (loss, component_dict)`.
* Baselines all implement `fit(train_loader)`, `score(x) -> unknown_score`,
  `predict(x) -> family_id | UNKNOWN`, sharing `data/splits.py` output and
  `eval/metrics.py`.

## Config-driven, reproducible by construction

Every run reads one YAML config (dataset + model + training + protocol), computes a
sha256 config hash, fixes `seed` for `random`/`numpy`/`torch`, and writes
`metrics.json` with: config hash, git commit, seed, package versions, hardware,
`is_synthetic` flag, and timing. `paper_artifacts` generation refuses any run where
`is_synthetic: true`.

See `docs/LRMC.md` for the loss math and `docs/DECISIONS.md` for design choices made
under ambiguity, and `HANDOFF.md` for the exact college-GPU-workstation run book
(Kaggle remains usable as an optional profile, see HANDOFF.md section 12).
