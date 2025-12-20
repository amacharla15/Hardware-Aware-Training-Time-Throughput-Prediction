import csv, statistics
from collections import defaultdict

TRAIN_N = 50000

g = defaultdict(list)

with open("runs.csv", newline="") as f:
    for r in csv.DictReader(f):
        k = (int(r["depth"]), int(r["base_filters"]), int(r["batch_size"]), int(r["params"]))
        g[k].append(r)

def stdev(v):
    if len(v) > 1:
        return statistics.stdev(v)
    return 0.0

rows_out = []

for (d, filt, bs, params), rows in g.items():
    t = []
    thr = []
    acc = []
    for x in rows:
        t.append(float(x["avg_time_sec"]))
        thr.append(float(x["images_per_sec"]))
        acc.append(float(x["test_acc"]))

    steps = (TRAIN_N + bs - 1) // bs

    rows_out.append({
        "depth": d,
        "base_filters": filt,
        "batch_size": bs,
        "params": params,
        "n": len(rows),
        "steps_per_epoch": steps,
        "avg_time_mean": statistics.mean(t),
        "avg_time_std": stdev(t),
        "thr_mean": statistics.mean(thr),
        "thr_std": stdev(thr),
        "acc_mean": statistics.mean(acc),
        "acc_std": stdev(acc),
    })

rows_out.sort(key=lambda r: (r["depth"], r["base_filters"], r["batch_size"]))

with open("runs_agg.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)

rep = []
for r in rows_out:
    if int(r["n"]) > 1:
        rep.append(r)

print("WROTE runs_agg.csv rows (unique configs):", len(rows_out))
print("configs_with_repeats (n>1):", len(rep))

if len(rep) > 0:
    rep.sort(key=lambda r: float(r["avg_time_std"]), reverse=True)
    print("top repeat-noise configs (by time std):")
    i = 0
    while i < 10 and i < len(rep):
        r = rep[i]
        print(r["depth"], r["base_filters"], r["batch_size"],
              "n=", r["n"],
              "time_std=", round(float(r["avg_time_std"]), 4),
              "thr_std=", round(float(r["thr_std"]), 1))
        i += 1
