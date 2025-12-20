import argparse
import math
import numpy as np
import tensorflow as tf

TRAIN_N = 50000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--base_filters", type=int, required=True)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--params", type=int, required=True)
    args = ap.parse_args()

    bs = int(args.batch_size)
    steps = (TRAIN_N + bs - 1) // bs

    X = []
    X.append([
        float(args.depth),
        float(args.base_filters),
        math.log2(float(args.batch_size)),
        math.log10(float(args.params)),
        math.log10(float(steps)),
    ])
    X = np.array(X, dtype=np.float32)

    model = tf.keras.models.load_model("results_v3/time_mean_ann_deploy.keras")
    pred_log_time = float(model.predict(X, verbose=0).reshape(-1)[0])

    pred_time = math.exp(pred_log_time)
    pred_thr = float(TRAIN_N) / pred_time

    print("predicted_avg_time_sec:", pred_time)
    print("predicted_images_per_sec_derived:", pred_thr)

if __name__ == "__main__":
    main()
