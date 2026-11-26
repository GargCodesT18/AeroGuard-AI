# Documentation asset provenance

Prepared on 2026-11-26. No model was trained and no numerical result was changed to create these assets.

## Dashboard screenshots

All eight supplied screenshots were located in the parent project's `docs/final-screenshots/` directory. No screenshot archive was required. Each supplied image was inspected visually and for metadata.

| Final file in `final-screenshots/` | Provenance and interpretation |
| --- | --- |
| `01-main-dashboard.png` | Fresh Edge capture of the local app after correcting the RUL/cap display label and 24-input count. Metrics are for FD001. |
| `02-prediction-healthy.png` | Supplied screenshot, copied with metadata removed; pixels unchanged. FD001 engine 1: 93.6 predicted cycles, 112 actual. |
| `03-prediction-critical.png` | Supplied screenshot, copied with metadata removed; pixels unchanged. FD001 engine 90: 22.0 predicted cycles, 28 actual. |
| `04-prediction-warning.png` | Supplied screenshot, copied with metadata removed; pixels unchanged. FD002 engine 17: 58.0 predicted cycles, 68 actual. |
| `05-model-insights.png` | Fresh Edge capture after correcting the latent dimension, target-range chart label and inherited architecture/framework footer. |
| `06-model-details.png` | Fresh Edge capture after correcting input shape, layer width, output activation/units, epoch count and illustrative profile labels. |
| `07-api-demo.png` | Fresh Edge capture of real local requests. Unsupported latency, node-count, model-version and throughput claims were replaced with measured request-pair time or accurate local-demo labels. |
| `08-data-explorer.png` | Fresh Edge capture after labelling the saved benchmark and final observations accurately. |

The five fresh captures use the existing dark-theme preference in Microsoft Edge, a 2560 × 1440 CSS viewport and device scale factor 0.75 (1920 × 1080 PNG output). They capture real DOM content and real CPU-backed Flask responses. Fonts and charts were allowed to finish loading. Images were not composited or numerically edited. Timing, MC intervals and small floating-point differences can vary with execution device and random draws.

The three copied prediction screenshots show the earlier CUDA-backed run. Their displayed point predictions agree with the saved prediction CSV to one decimal place; their interval draws need not match new CPU requests. Their originals have been retained outside this repository.

Rejected supplied screenshots were not copied into the final gallery: the original model-details image showed `[50, 21]`, a 128-unit first layer, Softmax and a [0, 1] output; the original insights image showed a 128-dimensional latent state; the API image claimed unmeasured infrastructure capacity. The supplied main/explorer images were superseded to correct misleading health/sensor/live-stream wording.

Historical upstream layout screenshots in `dashboard-shots/` remain unchanged for provenance. They contain outdated metrics and interface claims and are excluded from the README's current gallery. Placeholder profile labels and avatars in inherited pages are decorative, not evidence of authenticated accounts or commercial aircraft integrations.

## ML visualizations

Original artifacts remain in place. The following are copies of existing plots, with pixel equality verified after removing metadata:

| File in `ml-results/` | Source |
| --- | --- |
| `model-comparison.png` | `models/comparative/model_comparison.png` |
| `confusion-matrices.png` | `models/comparative/pooled_confusion_matrices.png` |
| `lstm-learning-curves.png` | `models/current/learning_curves.png` |
| `mlp-learning-curves.png` | `models/comparative/mlp_learning_curves.png` |
| `random-forest-feature-importance.png` | `models/comparative/random_forest_feature_importance.png` |
| `lstm-actual-vs-predicted-fd001.png` | `models/current/actual_vs_predicted_FD001.png` |
| `lstm-actual-vs-predicted-fd002.png` | `models/current/actual_vs_predicted_FD002.png` |
| `lstm-actual-vs-predicted-fd003.png` | `models/current/actual_vs_predicted_FD003.png` |
| `lstm-actual-vs-predicted-fd004.png` | `models/current/actual_vs_predicted_FD004.png` |

The selected current actual-versus-predicted originals existed locally as ignored evaluation outputs. Their copies are included here so README references resolve in a fresh clone.

Three additional figures were decoded from **existing executed PNG outputs** of `notebooks/AEROPREDICT_EDA.ipynb`, without executing its cells. Cell numbers below are zero-based:

| File in `ml-results/` | Notebook output |
| --- | --- |
| `fd001-engine-lifetimes.png` | Cell 10: training and observed test lifecycle distributions |
| `fd001-sensor-degradation.png` | Cell 12: selected sensor trends across normalized training life |
| `fd001-sensor-correlations.png` | Cell 13: settings/sensor correlation heatmap |

All 20 final PNGs were checked for EXIF and textual metadata and inspected visually for personal details. The 12 ML plots retain the scientific content of their sources. The executed notebooks retain every cell execution count and embedded scientific image/table; only local path text and its printing statements were sanitized.
