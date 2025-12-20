import os
import csv
import glob
import re
import subprocess
import shlex
import math

def load_runs_agg(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def find_params(rows, d, f):
    for r in rows:
        if int(r["depth"]) == d and int(r["base_filters"]) == f:
            return int(float(r["params"]))
    return None

def find_measured(rows, d, f, bs):
    for r in rows:
        if int(r["depth"]) == d and int(r["base_filters"]) == f and int(r["batch_size"]) == bs:
            return r
    return None

def run_predict(d, f, bs, p):
    cmd = f"python predict_perf_final.py --depth {d} --base_filters {f} --batch_size {bs} --params {p}"
    out = subprocess.check_output(shlex.split(cmd), text=True)
    t = None
    thr = None
    for line in out.splitlines():
        if line.startswith("predicted_avg_time_sec:"):
            t = float(line.split(":", 1)[1].strip())
        if line.startswith("predicted_images_per_sec_derived:"):
            thr = float(line.split(":", 1)[1].strip())
    return t, thr

def rel_err(pred, true):
    if true == 0:
        return float("nan")
    return abs(pred - true) / abs(true)

def main():
    rows = load_runs_agg("runs_agg.csv")

    files = sorted(glob.glob("logs_unseen/*.txt"))
    pat = re.compile(r"d(\d+)_f(\d+)_bs(\d+)\.txt$")

    configs = []
    for fp in files:
        m = pat.search(os.path.basename(fp))
        if not m:
            continue
        d = int(m.group(1))
        f = int(m.group(2))
        bs = int(m.group(3))
        configs.append((d, f, bs))

    if not configs:
        print("No logs_unseen/d*_f*_bs*.txt files found.")
        return

    print("Found unseen configs:", len(configs))

    time_abs = []
    thr_abs = []

    for (d, f, bs) in configs:
        p = find_params(rows, d, f)
        if p is None:
            print(f"\nCONFIG d={d} f={f} bs={bs} -> params NOT FOUND")
            continue

        meas = find_measured(rows, d, f, bs)
        if meas is None:
            print(f"\nCONFIG d={d} f={f} bs={bs} params={p} -> measured row NOT FOUND in runs_agg.csv")
            continue

        mt = float(meas["avg_time_mean"])
        mthr = float(meas["thr_mean"])
        n = int(meas["n"])

        pt, pthr = run_predict(d, f, bs, p)

        print(f"\nCONFIG d={d} f={f} bs={bs} params={p}  (n={n})")
        print(f"MEASURED  time={mt:.6f}  thr={mthr:.1f}")
        print(f"PREDICT   time={pt:.6f}  thr={pthr:.1f}")
        print(f"ABS ERR   time={abs(pt-mt):.6f}  thr={abs(pthr-mthr):.1f}")
        print(f"REL ERR   time={rel_err(pt, mt)*100:.2f}%  thr={rel_err(pthr, mthr)*100:.2f}%")

        time_abs.append(abs(pt - mt))
        thr_abs.append(abs(pthr - mthr))

    if time_abs:
        print("\nSUMMARY (over shown configs)")
        print("Time MAE:", sum(time_abs) / len(time_abs))
        print("Thr  MAE:", sum(thr_abs) / len(thr_abs))

if __name__ == "__main__":
    main()

