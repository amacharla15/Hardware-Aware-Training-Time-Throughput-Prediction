# Hardware-Aware Training Time & Throughput Prediction for CNNs on NVIDIA A100

This project builds a **hardware-aware performance predictor** for CNN training on **CIFAR-10** running on an **NVIDIA A100 GPU**.  
Given a CNN configuration (e.g., `depth`, `base_filters`, `batch_size`, `params`), the project predicts:

- **avg training time per epoch (sec/epoch)** *(main target)*
- **throughput (images/sec)** *(best method: derived as `50,000 / predicted_time_per_epoch` for CIFAR-10 train set)*

The core idea: collect real training runs on A100 → build a supervised regression dataset → train a regressor to predict time/epoch → derive throughput from time.

---

## Results (Final)

**Dataset (aggregated):**
- `runs_agg.csv`: **143 unique CNN configs** (aggregated from raw runs)

**Best time predictor (ANN, trained in log-space):**
- **MAE = 0.1014 sec/epoch**
- **RMSE = 0.1614 sec/epoch**
- **R² = 0.9938**

**Throughput strategy (derived from predicted time):**
- Throughput = `TRAIN_N / predicted_time`, where `TRAIN_N = 50,000`
- Derived throughput from ANN(time):  
  - **MAE = 2456.6 images/sec**
  - **RMSE = 3758.8 images/sec**
  - **R² = 0.9729**

**Generalization test (unseen configs):**
- 5 unseen configs in `logs_unseen/`
- Relative time errors: **~1.22% to 6.55%**
- Relative throughput errors: **~1.24% to 7.01%**
- Summary:
  - **Time MAE = 0.1039 sec/epoch**
  - **Throughput MAE = 1534.43 images/sec**

---

## Repository Layout (What Each File/Folder Is For)

### Phase 1 — Workload definition (CIFAR-10 CNN + timing)
- `starter_cnn_time.py`  
  Minimal baseline script: CIFAR-10 + simple CNN + epoch timing callback → prints `avg_time_per_epoch` and `images/sec`.
- `run_one.py`  
  Main configurable workload runner used for dataset collection (parameterized CNN by `depth`, `base_filters`, `batch_size`), logs results to CSV and writes a per-run log file.

### Phase 2 — Data collection (raw measurements)
- `runs.csv`  
  Raw dataset: **one row per run**.
- `logs/`, `logs_grid2/`, `logs_grid3/`, `logs_repeats/`, `logs_unseen/`  
  Full stdout logs for runs. Each contains a `LOGGED:` dictionary mirroring CSV fields.

### Phase 3 — Aggregation + feature engineering
- `make_runs_agg.py`  
  Groups repeated configs → computes mean/std → adds `steps_per_epoch`.
- `runs_agg.csv`  
  Final supervised-learning table (unique configs).

### Phase 4 — Visualization
- `make_plots_v2.py`  
  Generates plots from `runs_agg.csv`.
- `results_v2/`  
  Saved plot images (e.g., `acc_vs_time.png`, `throughput_vs_batch.png`, `repeats_time_errorbars.png`).

### Phase 5 — Time regressor training (ANN in log-space)
- `train_time_ann_deploy.py`  
  Trains ANN to predict **log(time/epoch)**; exponentiates during evaluation; saves deployable model.
- `results_v3/time_mean_ann_deploy.keras`  
  Saved deployable ANN model.

### Phase 6 — Throughput prediction strategy
- `train_time_derive_thr.py`  
  Evaluates derived throughput using `thr = 50,000 / predicted_time`, compares Linear(time) vs ANN(time).
- `results_v3/thr_derived_from_time_ann.png`  
- `results_v3/thr_derived_from_time_linear.png`  
- `results_v3/derived_thr_report_latest.txt` *(if present)*

### Phase 7 — Generalization test (unseen configs)
- `eval_unseen.py`  
  Loads unseen configs from `logs_unseen/`, compares measured vs predicted time & throughput, prints MAE/errors.

### Phase 8 — Time-budget planner (Dynamic Programming / 0/1 Knapsack)
- `dp_budget_planner.py`  
  Selects the best set of candidate CNN runs under a time budget.
  - cost = `round(time_per_epoch * epochs)`
  - value = `acc_mean` or `thr_mean`
  - DP state: `dp[t] = best value within budget t`
- `results_v3/dp_plan_*.txt` *(generated outputs)*

### Phase 9 — Deployment (final inference interface)
- `predict_perf_final.py`  
  Loads saved ANN model → predicts time/epoch for a new config → derives throughput.

---

## Data Schema

### `runs.csv` (raw, per run)
Columns:
- `timestamp_utc`, `tf_version`
- `depth`, `base_filters`, `batch_size`, `epochs`
- `params`
- `epoch_times_json`
- `avg_time_sec`
- `images_per_sec`
- `test_acc`

### `runs_agg.csv` (aggregated, per unique config)
Columns (typical):
- `depth`, `base_filters`, `batch_size`, `params`, `n`
- `steps_per_epoch`
- `avg_time_mean`, `avg_time_std`
- `thr_mean`, `thr_std`
- `acc_mean`, `acc_std`

---

## How the Model Works (Feature Engineering)

For each configuration, the time regressor uses:

- `depth`
- `base_filters`
- `log2(batch_size)`
- `log10(params)`
- `log10(steps_per_epoch)`

Where:
- `steps_per_epoch = ceil(50,000 / batch_size)` (CIFAR-10 fixed train set)

**Why log-space for the target?**
- Time is positive and spans orders of magnitude.
- Training in log-space prevents invalid negative time predictions.
- Prediction pipeline:
  - model outputs `log(time)`
  - final `time = exp(pred_log_time)`

---

## Phase-Based Pipeline (End-to-End)

### Phase 1 — Define the workload
Goal: a repeatable timing workload generator on CIFAR-10.

### Phase 2 — Collect raw runs on A100
Goal: build `runs.csv` and logs folders as ground truth.

### Phase 3 — Aggregate + derive features
Goal: reduce noise + produce `runs_agg.csv` (mean/std + `steps_per_epoch`).

### Phase 4 — Visualize patterns
Goal: show relationships like accuracy vs time, throughput vs batch, timing noise via repeats.

### Phase 5 — Train time/epoch predictor
Goal: supervised ANN model predicting time/epoch from config features (saved model).

### Phase 6 — Derive throughput from predicted time
Goal: stable throughput prediction using physics/definition: `50,000 / time_per_epoch`.

### Phase 7 — Unseen generalization test
Goal: validate model on configs not used in the main training grid.

### Phase 8 — Dynamic programming planner
Goal: select configs to run under limited time budgets (0/1 knapsack DP).

### Phase 9 — Deployment interface
Goal: simple script to predict performance for new configs.

---

## Commands (Quick, Minimal)

> Note: Full training/data collection was done on the A100 environment.  
> The commands below show the intended pipeline.

### 1) Aggregate raw runs → `runs_agg.csv`

python3 make_runs_agg.py
2) Generate plots → results_v2/
python3 make_plots_v2.py

3) Train & save time model → results_v3/time_mean_ann_deploy.keras
python3 train_time_ann_deploy.py

4) Evaluate derived throughput strategy
python3 train_time_derive_thr.py

5) Evaluate unseen configs
python3 eval_unseen.py

6) DP planner example (budgeted selection)
python3 dp_budget_planner.py --budget_sec 3600 --epochs 50 --min_acc 0.60 --value acc --time_source measured

7) Predict performance for a new config
python3 predict_perf_final.py --depth 3 --base_filters 32 --batch_size 512 --params 550570

Environment / Dependencies

Typical stack used:

Python 3.x

TensorFlow/Keras (used during training + inference)

NumPy

scikit-learn

matplotlib

Hardware used for data collection:

NVIDIA A100 GPU (CSU Chico cscigpu)

Notes / Design Choices

Why CIFAR-10? Fixed training size (50,000) makes throughput derivation consistent and comparable.

Why exclude epoch 1 for timing? First epoch is often slower due to warmup/autotune/caching effects.

Why aggregate repeats? Timing noise exists even for identical configs; mean/std produces stable targets.

Ethics & Responsible Use

This model predicts training performance for CNN configurations on a specific hardware/software setup (A100 + TF stack).
Predictions may not transfer to different GPUs, drivers, mixed precision settings, or dataloader pipelines without new measurements.

Acknowledgments / References

CIFAR-10 dataset (TensorFlow tf.keras.datasets.cifar10)

TensorFlow/Keras for training and model deployment

```bash
