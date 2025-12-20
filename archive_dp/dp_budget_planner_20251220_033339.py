import argparse
import math
import os
import sys

import numpy as np
import pandas as pd


def build_feature_vector(d, bf, bs, params, steps_per_epoch):
    # Must match train_time_ann_deploy.py / predict_perf_final.py
    # Features = [depth, base_filters, log2(batch_size), log10(params), log10(steps_per_epoch)]
    x = np.zeros((1, 5), dtype=np.float32)
    x[0, 0] = float(d)
    x[0, 1] = float(bf)

    # safe logs
    bsv = float(bs)
    pv = float(params)
    sv = float(steps_per_epoch)
    if bsv <= 0.0:
        bsv = 1.0
    if pv <= 0.0:
        pv = 1.0
    if sv <= 0.0:
        sv = 1.0

    x[0, 2] = float(math.log2(bsv))
    x[0, 3] = float(math.log10(pv))
    x[0, 4] = float(math.log10(sv))
    return x

def predict_time_per_epoch_sec(model, d, bf, bs, params, steps_per_epoch):
    x = build_feature_vector(d, bf, bs, params, steps_per_epoch)
    y = model.predict(x, verbose=0)

    if isinstance(y, (list, tuple)):
        y = np.array(y)

    pred_log_time = float(np.ravel(y)[0])

    # convert log(time) -> time
    if not math.isfinite(pred_log_time):
        pred_log_time = 0.0

    pred_time = float(math.exp(pred_log_time))

    # clamp
    if (not math.isfinite(pred_time)) or pred_time <= 0.0:
        pred_time = 1e-6

    return pred_time


def load_time_model(model_path, cpu_only):
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    return tf.keras.models.load_model(model_path, compile=False)




def knapsack_01(items, budget_sec):
    # 0/1 knapsack with correct reconstruction using 2D DP
    B = int(budget_sec)
    N = len(items)

    dp = []
    take = []

    i = 0
    while i <= N:
        dp.append([0.0] * (B + 1))
        take.append([0] * (B + 1))
        i += 1

    i = 1
    while i <= N:
        c = int(items[i - 1]["cost"])
        v = float(items[i - 1]["value"])

        t = 0
        while t <= B:
            best = dp[i - 1][t]
            chosen = 0

            if c > 0 and t >= c:
                cand = dp[i - 1][t - c] + v
                if cand > best:
                    best = cand
                    chosen = 1

            dp[i][t] = best
            take[i][t] = chosen
            t += 1

        i += 1

    best_t = 0
    best_v = dp[N][0]
    t = 1
    while t <= B:
        if dp[N][t] > best_v:
            best_v = dp[N][t]
            best_t = t
        t += 1

    chosen_items = []
    t = best_t
    i = N
    while i >= 1:
        if take[i][t] == 1:
            chosen_items.append(items[i - 1])
            t -= int(items[i - 1]["cost"])
        i -= 1

    return chosen_items, best_t, best_v



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_agg", default="runs_agg.csv")
    ap.add_argument("--model", default="results_v3/time_mean_ann_deploy.keras")
    ap.add_argument("--time_source", choices=["measured", "pred"], default="measured")
    ap.add_argument("--cpu_only", action="store_true")
    ap.add_argument("--value", choices=["acc", "thr"], default="acc")
    ap.add_argument("--min_acc", type=float, default=0.0)
    ap.add_argument("--budget_sec", type=int, default=3600)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--max_items", type=int, default=0, help="0 = no limit; else cap candidates")
    args = ap.parse_args()

    if not os.path.exists(args.runs_agg):
        print("ERROR: runs_agg.csv not found:", args.runs_agg)
        sys.exit(1)

    df = pd.read_csv(args.runs_agg)

    needed = ["depth", "base_filters", "batch_size", "params", "steps_per_epoch",
              "avg_time_mean", "thr_mean", "acc_mean"]
    for c in needed:
        if c not in df.columns:
            print("ERROR: missing column in runs_agg.csv:", c)
            sys.exit(1)

    df = df.copy()
    df = df[(df["depth"] > 0) & (df["base_filters"] > 0) & (df["batch_size"] > 0) &
            (df["params"] > 0) & (df["steps_per_epoch"] > 0)]

    df = df[df["acc_mean"] >= float(args.min_acc)].copy()

    model = None
    if args.time_source == "pred":
        if not os.path.exists(args.model):
            print("ERROR: time model not found:", args.model)
            sys.exit(1)
        model = load_time_model(args.model, args.cpu_only)

    items = []
    i = 0
    while i < len(df):
        r = df.iloc[i]
        d = int(r["depth"])
        bf = int(r["base_filters"])
        bs = int(r["batch_size"])
        params = int(r["params"])
        steps = float(r["steps_per_epoch"])

        acc = float(r["acc_mean"])
        thr = float(r["thr_mean"])
        t_meas = float(r["avg_time_mean"])

        if args.time_source == "measured":
            time_per_epoch = t_meas
        else:
            time_per_epoch = predict_time_per_epoch_sec(model, d, bf, bs, params, steps)

        cost = int(round(time_per_epoch * float(args.epochs)))
        if cost <= 0:
            i += 1
            continue
        if cost > args.budget_sec:
            i += 1
            continue

        if args.value == "acc":
            val = acc
        else:
            val = thr

        items.append({
            "depth": d,
            "base_filters": bf,
            "batch_size": bs,
            "params": params,
            "steps_per_epoch": steps,
            "acc": acc,
            "thr": thr,
            "time_per_epoch": time_per_epoch,
            "epochs": int(args.epochs),
            "cost": int(cost),
            "value": float(val),
        })
        i += 1

    if args.max_items and args.max_items > 0 and len(items) > args.max_items:
        items.sort(key=lambda it: (it["value"] / max(1, it["cost"])), reverse=True)
        items = items[: args.max_items]

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
    print(f"Budget utilization: {100.0 * float(total_cost) / float(args.budget_sec):.2f}%")
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
