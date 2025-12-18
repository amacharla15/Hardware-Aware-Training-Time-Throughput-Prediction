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


TRAIN_N = 50000


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def build_x(rows):
    X = []
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
    return np.array(X, dtype=np.float32)


def build_y_time_log(rows):
    y = []
    for r in rows:
        t = float(r["avg_time_mean"])
        if t <= 0.0:
            t = 1e-12
        y.append(math.log(t))
    return np.array(y, dtype=np.float32)


def build_y_thr(rows):
    y = []
    for r in rows:
        y.append(float(r["thr_mean"]))
    return np.array(y, dtype=np.float32)


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


def main():
    os.makedirs("results_v3", exist_ok=True)

    rows = load_rows("runs_agg.csv")
    print("rows:", len(rows))

    X = build_x(rows)
    y_time_log = build_y_time_log(rows)
    y_thr = build_y_thr(rows)

    X_train, X_temp, y_train, y_temp, thr_train, thr_temp = train_test_split(
        X, y_time_log, y_thr, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test, thr_val, thr_test = train_test_split(
        X_temp, y_temp, thr_temp, test_size=0.50, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # -------- Linear time model (on log time) --------
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    pred_time_log_lr = lr.predict(X_test_s)
    pred_time_lr = np.exp(pred_time_log_lr)

    pred_thr_from_lr = TRAIN_N / pred_time_lr
    mae, rmse, r2 = eval_metrics(thr_test, pred_thr_from_lr)
    print(f"DERIVED thr_mean from Linear(time)  MAE={mae:.6f}  RMSE={rmse:.6f}  R2={r2:.6f}")
    save_scatter(thr_test, pred_thr_from_lr, "thr_mean derived from Linear(time)", "results_v3/thr_derived_from_time_linear.png")

    # -------- ANN time model (on log time) --------
    tf.random.set_seed(42)
    np.random.seed(42)

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

    pred_time_log_ann = model.predict(X_test_s, verbose=0).reshape(-1)
    pred_time_ann = np.exp(pred_time_log_ann)

    pred_thr_from_ann = TRAIN_N / pred_time_ann
    mae, rmse, r2 = eval_metrics(thr_test, pred_thr_from_ann)
    print(f"DERIVED thr_mean from ANN(time)     MAE={mae:.6f}  RMSE={rmse:.6f}  R2={r2:.6f}")
    save_scatter(thr_test, pred_thr_from_ann, "thr_mean derived from ANN(time)", "results_v3/thr_derived_from_time_ann.png")


if __name__ == "__main__":
    main()
