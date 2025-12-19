# Hardware-Aware Training Time & Throughput Prediction (A100, CIFAR-10)

This project collects training-time/throughput measurements for many CNN configurations on an NVIDIA A100 GPU (CIFAR-10),
then trains regressors (Linear + ANN) to predict performance from config features.

Key idea: throughput is best predicted by predicting time and deriving throughput:
images/sec ≈ 50000 / time_per_epoch.

## Files
- runs.csv: raw run logs (one row per training job)
- make_runs_agg.py -> runs_agg.csv: aggregates repeats + derives steps_per_epoch
- train_time_ann_deploy.py: trains deployable ANN time model (log-target) -> results_v3/time_mean_ann_deploy.keras
- predict_perf_final.py: loads deploy model and predicts time + derived throughput
- make_plots_v2.py: plots in results_v2/
