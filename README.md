# Hardware-Aware Training Time + Throughput Prediction (CNNs on A100)

Measures CNN training performance on CIFAR-10 on an NVIDIA A100 and trains regressors to predict:
- avg time per epoch (seconds)
- throughput (images/sec), derived from predicted time

## Key files
- `run_one.py`: run one config + log to `runs.csv`
- `make_runs_agg.py` -> `runs_agg.csv`: aggregates repeats (mean/std) + steps_per_epoch
- `train_regressor*.py`: Linear + ANN regressors (analysis)
- `train_time_ann_deploy.py`: deployable ANN time model (log-space)
- `predict_perf_final.py`: final CLI predictor (time + derived throughput)

## Quick start
```bash
source venv/bin/activate
python predict_perf_final.py --depth 2 --base_filters 16 --batch_size 2048 --params 280218
