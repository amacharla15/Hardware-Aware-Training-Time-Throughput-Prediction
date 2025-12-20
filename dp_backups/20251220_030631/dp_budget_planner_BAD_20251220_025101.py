import argparse
import csv
import math
import os
import sys

import numpy as np


def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def load_time_model(model_path):
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def build_feature_vector(d, bf, bs, params, steps_per_epoch):
    x = np.zeros((1, 5), dtype=np.float32)
    x[0, 0] = float(d)
    x[0, 1] = float(bf)
    x[0, 2] = float(bs)
    x[0, 3] = float(params)
    x[0, 4] = float(steps_per_epoch)
    return x


def predict_time_per_epoch_sec(model, d, bf, bs, params, steps_per_epoch):
    x = build_feature_vector(d, bf, bs, params, steps_per_epoch)
    y = model.predict(x, verbose=0)
    yv = float(np.ravel(y)[0])
    if (not math.isfinite(yv)) or (yv <= 0.0):
        yv = 1e-6
    return yv


def read_runs_agg(runs_agg_path):
    rows = []
    with open(runs_agg_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def knapsack_01(items, budget_sec):
    B = int(budget_sec)
    neg_inf = -1e30

    dp = [neg_inf] * (B + 1)
    take = [-1] * (B + 1)
    prev = [-1] * (B + 1)

    dp[0] = 0.0

    i = 0
    while i < len(items):
        c = int(items[i]["cost"])
        v = float(items[i]["value"])
        if c <= 0:
            i += 1
            continue
        t = B
        while t >= c:
            if dp[t - c] > neg_inf / 2:
                cand = dp[t - c] + v
                if cand > dp[t]:
                    dp[t] = cand
                    take[t] = i
                    prev[t] = t - c
            t -= 1
        i += 1

    best_t = 0
    best_v = dp[0]
    t = 1
    while t <= B:
        if dp[t] > best_v:
            best_v = dp[t]
            best_t = t
        t += 1

    chosen = []
    t = best_t
    used = set()
    while t >= 0 and take[t] != -1:
        i = take[t]
        if i in used:
            break
        used.add(i)
        chosen.append(items[i])
        t = prev[t]
        if t < 0:
            break

    return chosen, best_t, best_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_agg", default="runs_agg.csv")
    ap.add_argument("--budget_sec", type=int, default=3600)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--min_acc", type=float, default=0.0)

    ap.add_argument("--value", choices=["acc", "thr"], default="acc")
    ap.add_argument("--time_source", choices=["measured", "pred"], default="measured")
    ap.add_argument("--model", default="results_v3/time_mean_ann_deploy.keras")

    ap.add_argument("--max_items", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.runs_agg):
        print("ERROR: runs_agg.csv not found at", args.runs_agg)
        sys.exit(1)

    model = None
    if args.time_source == "pred":
        if not os.path.exists(args.model):
            print("ERROR: time model not found at", args.model)
            sys.exit(1)
        model = load_time_model(args.model)

    agg_rows = read_runs_agg(args.runs_agg)

    items = []
    for row in agg_rows:
        d = safe_int(row.get("depth", 0))
        bf = safe_int(row.get("base_filters", 0))
        bs = safe_int(row.get("batch_size", 0))
        params = safe_int(row.get("params", 0))
        steps = safe_float(row.get("steps_per_epoch", 0.0))

        if d <= 0 or bf <= 0 or bs <= 0 or params <= 0 or steps <= 0:
            continue

        acc = safe_float(row.get("acc_mean", 0.0))
        thr = safe_float(row.get("thr_mean", 0.0))
        t_meas = safe_float(row.get("avg_time_mean", 0.0))

        if acc < args.min_acc:
            continue

        if args.time_source == "measured":
            time_per_epoch = t_meas
        else:
            time_per_epoch = predict_time_per_epoch_sec(model, d, bf, bs, params, steps)

        if (not math.isfinite(time_per_epoch)) or (time_per_epoch <= 0.0):
            continue

        total_cost_sec = int(round(time_per_epoch * float(args.epochs)))
        if total_cost_sec <= 0:
            total_cost_sec = 1

        if total_cost_sec > args.budget_sec:
            continue

        if args.value == "acc":
            value = acc
        else:
            value = thr

        item = {
            "depth": d,
            "base_filters": bf,
            "batch_size": bs,
            "params": params,
            "steps_per_epoch": steps,
            "acc": acc,
            "thr": thr,
            "time_per_epoch": time_per_epoch,
            "epochs": args.epochs,
            "cost": total_cost_sec,
            "value": value,
        }
        items.append(item)

    if args.max_items and args.max_items > 0 and len(items) > args.max_items:
        items.sort(key=lambda it: (it["value"] / max(1, it["cost"])), reverse=True)
        items = items[: args.max_items]

    items.sort(key=lambda it: (it["value"], -it["cost"]), reverse=True)

    print("DP TIME-BUDGET PLANNER (0/1 Knapsack)")
    print(f"Budget: {args.budget_sec} sec  (~{args.budget_sec/60.0:.2f} min)")
    print(f"Epochs per candidate run: {args.epochs}")
    print(f"Value optimized: {args.value}")
    print(f"Time source: {args.time_source}")
    print(f"min_acc filter: {args.min_acc}")
    print(f"Candidates available: {len(items)}")

    if len(items) == 0:
        print("No candidates fit the budget (or min_acc removed all).")
        sys.exit(0)

    chosen, used_t, best_v = knapsack_01(items, args.budget_sec)

    total_cost = 0
    total_value = 0.0
    i = 0
    while i < len(chosen):
        total_cost += int(chosen[i]["cost"])
        total_value += float(chosen[i]["value"])
        i += 1

    chosen.sort(key=lambda it: it["cost"])

    print("")
    print("RESULT")
    print(f"Chosen configs: {len(chosen)}")
    print(f"Total time used: {total_cost} sec  (~{total_cost/60.0:.2f} min)")
    print(f"Budget utilization: {100.0*float(total_cost)/float(args.budget_sec):.2f}%")
    print(f"Total value: {total_value:.6f}")
    print("")
    print("Chosen list:")
    print("depth  filters  batch   params      acc     thr(img/s)  time/ep(s)  epochs  cost(s)")

    i = 0
    while i < len(chosen):
        it = chosen[i]
        print(
            f"{it['depth']:5d}  {it['base_filters']:7d}  {it['batch_size']:5d}  {it['params']:9d}  "
            f"{it['acc']:7.4f}  {it['thr']:10.1f}  {it['time_per_epoch']:10.4f}  {it['epochs']:5d}  {it['cost']:6d}"
        )
        i += 1

    print("")
    print("DP details (for report):")
    print("- 0/1 knapsack DP (each candidate config can be chosen at most once).")
    print("- cost = time_per_epoch * epochs (rounded to seconds).")
    print("- value = acc_mean (or thr_mean) from runs_agg.csv.")
    print("- state: dp[t] = best total value achievable within time budget t.")
    print("- transition: dp[t] = max(dp[t], dp[t-cost_i] + value_i).")
    print(f"- complexity: O(N*B) with N={len(items)} and B={args.budget_sec} seconds.")


if __name__ == "__main__":
    main()
