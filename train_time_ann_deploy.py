import csv
import math
import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    i = 0
    while i < len(rows):
        r = rows[i]
        depth = float(r["depth"])
        base_filters = float(r["base_filters"])
        batch_size = float(r["batch_size"])
        params = float(r["params"])
        steps = float(r["steps_per_epoch"])

        X.append([
            depth,
            base_filters,
            math.log2(batch_size),
            math.log10(params),
            math.log10(steps),
        ])

        t = float(r[target_key])
        if t <= 0.0:
            t = 1e-9
        y.append(math.log(t))

        i += 1

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def eval_time_metrics(y_log_true, y_log_pred):
    y_true = np.exp(y_log_true)
    y_pred = np.exp(y_log_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def main():
    os.makedirs("results_v3", exist_ok=True)

    rows = load_rows("runs_agg.csv")
    print("rows:", len(rows))

    X, y_log = build_xy(rows, "avg_time_mean")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y_log, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    norm = tf.keras.layers.Normalization(axis=-1)
    norm.adapt(X_train)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        norm,
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=500,
        batch_size=32,
        verbose=0,
        callbacks=[es],
    )

    pred_log = model.predict(X_test, verbose=0).reshape(-1)

    mae, rmse, r2 = eval_time_metrics(y_test, pred_log)
    print(f"time_mean_ANN_DEPLOY  MAE={mae:.6f}  RMSE={rmse:.6f}  R2={r2:.6f}")

    model.save("results_v3/time_mean_ann_deploy.keras")
    print("saved results_v3/time_mean_ann_deploy.keras")


if __name__ == "__main__":
    main()
