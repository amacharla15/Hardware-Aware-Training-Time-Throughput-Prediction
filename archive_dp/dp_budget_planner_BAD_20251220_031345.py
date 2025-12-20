import argparse
import csv
import math
import os
import sys


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def read_runs_agg_rows(runs_agg_path):
    rows = []
    with open(runs_agg_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_feature_vector_like_train(d, bf, bs, params, steps_per_epoch):
    # MUST match train_time_ann_deploy.py / predict_perf_final.py
    # Features were:
    #   depth, base_filters, log2(batch_size), log10(params), log10(steps_per_epoch)
    x0 = float(d)
    x1 = float(bf)
    x2 = math.log2(float(bs))
    x3 = math.log10(float(params))
    x4 = math.log10(float(steps_per_epoch))
    return [[x0, x1, x2, x3, x4]]


def maybe_force_cpu(cpu_only):
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def load_time_model(model_path, cpu_only):
    maybe_force_cpu(cpu_only)
    import numpy as np
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    return model, np


def predict_time_per_epoch_sec(model, np_mod, d, bf, bs, params, steps_per_epoch):
    X = build_feature_vector_like_train(d, bf, bs, params, steps_per_epoch)
    X = np_mod.array(X, dtype=np_mod.float32)

    pred_log_time = float(model.predict(X, verbose=0).reshape(-1)[0])
    pred_time = math.exp(pred_log_time)

    if (not math.isfinite(pred_time)) or pred_time <= 0.0:
        pred_time = 1e-6
    return pred_time


def knapsack_01(items, budget_sec):
    # 0/1 knapsack with integer-second budget
    B = int(budget_sec)

    dp = [-1e30] * (B + 1)
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

        # IMPORTANT: iterate time backwards for 0/1 (prevents choosing same item multiple times)
        t = B
        while t >= c:
            if dp[t - c] > -1e29:
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
    used = set()
    t = best_t
    while t >= 0 and take[t] != -1:
        i = take[t]
        if i in used:
            break
        used.add(i)
        chosen.append(items[i])
        t = prev[t]

    return chosen, best_t, best_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_agg", default="runs_agg.csv")
    ap.add_argument("--model", default="results_v3/time_mean_ann_deploy.keras")

    ap.add_argument("--time_source", choices=["measured", "pred"], default="measured")
    ap.add_argument("--cpu_only", action="store_true", help="If time_source=pred, load TF model on CPU (less GPU noise)")

    ap.add_argument("--value", choices=["acc", "thr"], default="acc")
    ap.add_argument("--min_acc", type=float, default=0.0)

    ap.add_argument("--budget_sec", type=int, default=3600)
    ap.add_argument("--epochs", type=int, default=3)

    ap.add_argument("--max_items", type=int, default=0, help="0=no cap. Else keep top-K by value density for faster DP.")
    args = ap.parse_args()

    if not os.path.exists(args.runs_agg):
        print("ERROR: runs_agg.csv not found at", args.runs_agg)
        sys.exit(1)

    if args.time_source == "pred" and (not os.path.exists(args.model)):
        print("ERROR: time model not found at", args.model)
        sys.exit(1)

    model = None
    np_mod = None
    if args.time_source == "pred":
        model, np_mod = load_time_model(args.model, args.cpu_only)

    agg_rows = read_runs_agg_rows(args.runs_agg)

    # Build candidate items (dedupe by config just in case)
    seen = set()
    items = []

    for row in agg_rows:
        d = safe_int(row.get("depth", 0))
        bf = safe_int(row.get("base_filters", 0))
        bs = safe_int(row.get("batch_size", 0))
        params = safe_int(row.get("params", 0))
        steps = safe_float(row.get("steps_per_epoch", 0.0))

        acc = safe_float(row.get("acc_mean", 0.0))
        thr = safe_float(row.get("thr_mean", 0.0))
        t_meas = safe_float(row.get("avg_time_mean", 0.0))

        if d <= 0 or bf <= 0 or bs <= 0 or params <= 0 or steps <= 0:
            continue

        if acc < args.min_acc:
            continue

        key = (d, bf, bs, params)
        if key in seen:
            continue
        seen.add(key)

        if args.time_source == "measured":
            time_per_epoch = t_meas
        else:
            time_per_epoch = predict_time_per_epoch_sec(model, np_mod, d, bf, bs, params, steps)

        if (not math.isfinite(time_per_epoch)) or time_per_epoch <= 0.0:
            continue

        total_cost_sec = int(round(time_per_epoch * float(args.epochs)))
        if total_cost_sec <= 0:
            total_cost_sec = 1

        if total_cost_sec > args.budget_sec:
            continue

        if args.value == "acc":
            val = acc
        else:
            val = thr

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
            "value": val,
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
    for it in chosen:
        total_cost += int(it["cost"])
        total_value += float(it["value"])

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
    for it in chosen:
        print(f"{it['depth']:5d}  {it['base_filters']:7d}  {it['batch_size']:5d}  {it['params']:9d}  "
              f"{it['acc']:7.4f}  {it['thr']:10.1f}  {it['time_per_epoch']:10.4f}  {it['epochs']:5d}  {it['cost']:6d}")

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
