import csv
import math
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

import tensorflow as tf


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def build_xy(rows, target_key):
    X = []
    y = []
    for r in rows:
        depth = float(r["depth"])
        base_filters = float(r["base_filters"])
        batch_size = float(r["batch_size"])
        params = float(r["params"])
        steps_per_epoch = float(r["steps_per_epoch"])

        X.append([
            depth,
            base_filters,
            math.log2(batch_size),
            math.log10(params),
            math.log2(steps_per_epoch),
        ])
        y.append(float(r[target_key]))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def save_scatter(y_true, y_pred, title, outpath):
    plt.figure()
    plt.scatter(y_true, y_pred)
    mn = min(float(np.min(y_true)), float(np.min(y_pred)))
    mx = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def train_target(rows, target_key, prefix, outdir):
    X, y = build_xy(rows, target_key)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Linear baseline
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    pred_lr = lr.predict(X_test_s)
    mae, rmse, r2 = eval_metrics(y_test, pred_lr)
    print(f"{target_key} LinearRegression  MAE={mae:.6f}  RMSE={rmse:.6f}  R2={r2:.6f}")
    save_scatter(y_test, pred_lr, f"{target_key} LinearRegression", f"{outdir}/{prefix}_linear_scatter.png")

    # ANN regressor (same model family)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_s.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=200,
        batch_size=32,
        verbose=0,
        callbacks=[es],
    )

    pred_ann = model.predict(X_test_s, verbose=0).reshape(-1)
    mae, rmse, r2 = eval_metrics(y_test, pred_ann)
    print(f"{target_key} ANN               MAE={mae:.6f}  RMSE={rmse:.6f}  R2={r2:.6f}")
    save_scatter(y_test, pred_ann, f"{target_key} ANN", f"{outdir}/{prefix}_ann_scatter.png")
    model.save(f"{outdir}/{prefix}_ann_model.keras")


def main():
    outdir = "results_v2"
    os.makedirs(outdir, exist_ok=True)

    rows = load_rows("runs_agg.csv")
    print("rows:", len(rows))

    train_target(rows, "avg_time_mean", "time_mean", outdir)
    train_target(rows, "thr_mean", "throughput_mean", outdir)


if __name__ == "__main__":
    main()

