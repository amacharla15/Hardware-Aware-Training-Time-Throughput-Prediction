import argparse
import csv
import json
import os
import time
from datetime import datetime
import tensorflow as tf

class EpochTimer(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []
    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.perf_counter()
    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self.t0)

def make_model(depth, base_filters):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 32, 3)))

    filters = base_filters
    i = 0
    while i < depth:
        model.add(tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu"))
        model.add(tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu"))
        model.add(tf.keras.layers.MaxPool2D())
        filters = filters * 2
        i += 1

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--base_filters", type=int, required=True)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--csv_path", type=str, default="runs.csv")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.cache().shuffle(50000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.cache().batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = make_model(args.depth, args.base_filters)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    timer = EpochTimer()
    history = model.fit(train_ds, epochs=args.epochs, callbacks=[timer], verbose=2)

    epoch_times = timer.times
    if len(epoch_times) >= 2:
        avg_time = 0.0
        i = 1
        while i < len(epoch_times):
            avg_time += epoch_times[i]
            i += 1
        avg_time = avg_time / (len(epoch_times) - 1)
    else:
        avg_time = epoch_times[0]

    images_per_sec = float(x_train.shape[0]) / float(avg_time)

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    row = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "tf_version": tf.__version__,
        "depth": args.depth,
        "base_filters": args.base_filters,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "params": int(model.count_params()),
        "epoch_times_json": json.dumps(epoch_times),
        "avg_time_sec": float(avg_time),
        "images_per_sec": float(images_per_sec),
        "test_acc": float(test_acc),
    }

    write_header = not os.path.exists(args.csv_path) or os.path.getsize(args.csv_path) == 0
    with open(args.csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print("LOGGED:", row)

if __name__ == "__main__":
    main()
