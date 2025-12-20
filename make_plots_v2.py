import csv
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("runs_agg.csv", newline="")))

time = []
acc = []
thr = []
bs = []

for r in rows:
    time.append(float(r["avg_time_mean"]))
    acc.append(float(r["acc_mean"]))
    thr.append(float(r["thr_mean"]))
    bs.append(int(r["batch_size"]))

import os
os.makedirs("results_v2", exist_ok=True)

# (A) Accuracy vs Time
plt.figure()
plt.scatter(time, acc)
plt.xlabel("avg_time_per_epoch_sec (mean)")
plt.ylabel("test_acc (mean)")
plt.title("CIFAR-10 Accuracy vs Training Time per Epoch (A100)")
plt.tight_layout()
plt.savefig("results_v2/acc_vs_time.png", dpi=200)
plt.close()

# (B) Throughput vs Batch Size
plt.figure()
plt.scatter(bs, thr)
plt.xscale("log")
plt.xlabel("batch_size (log scale)")
plt.ylabel("images_per_sec (mean)")
plt.title("Throughput vs Batch Size (A100)")
plt.tight_layout()
plt.savefig("results_v2/throughput_vs_batch.png", dpi=200)
plt.close()

# (C) Repeat timing error bars
rep = []
for r in rows:
    if int(r["n"]) > 1:
        rep.append(r)

if len(rep) > 0:
    labels = []
    y = []
    yerr = []
    for r in rep:
        labels.append("d{}_f{}_bs{}".format(r["depth"], r["base_filters"], r["batch_size"]))
        y.append(float(r["avg_time_mean"]))
        yerr.append(float(r["avg_time_std"]))

    plt.figure()
    plt.errorbar(range(len(rep)), y, yerr=yerr, fmt="o")
    plt.xticks(range(len(rep)), labels, rotation=45, ha="right")
    plt.ylabel("avg_time_per_epoch_sec (mean ± std)")
    plt.title("Run-to-run Timing Variance (Repeated Configs)")
    plt.tight_layout()
    plt.savefig("results_v2/repeats_time_errorbars.png", dpi=200)
    plt.close()

print("saved plots to results_v2/")
