import time
import tensorflow as tf

class EpochTimer(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []
    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.perf_counter()
    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self.t0)

def main():
    print("TF version:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    batch_size = 128
    epochs = 3

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(50000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.summary()
    print("count_params:", model.count_params())
    print("batch_size:", batch_size)
    steps_per_epoch = (x_train.shape[0] + batch_size - 1) // batch_size
    print("steps_per_epoch:", steps_per_epoch)

    timer = EpochTimer()
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[timer], verbose=2)

    times = timer.times
    print("epoch_times_sec:", times)

    if len(times) >= 2:
        avg = sum(times[1:]) / (len(times) - 1)
    else:
        avg = times[0]

    images_per_sec = x_train.shape[0] / avg
    print("avg_time_per_epoch_sec (excluding epoch1):", avg)
    print("images_per_sec:", images_per_sec)

if __name__ == "__main__":
    main()
