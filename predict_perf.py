import argparse
import math
import numpy as np
import tensorflow as tf

def build_features(depth, base_filters, batch_size, params):
    x0 = float(depth)
    x1 = float(base_filters)
    x2 = math.log2(float(batch_size))
    x3 = math.log10(float(params))
    return np.array([[x0, x1, x2, x3]], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--base_filters", type=int, required=True)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--params", type=int, required=True)
    ap.add_argument("--time_model", type=str, default="results/time_ann_model.keras")
    ap.add_argument("--thr_model", type=str, default="results/throughput_ann_model.keras")
    args = ap.parse_args()

    X = build_features(args.depth, args.base_filters, args.batch_size, args.params)

    time_model = tf.keras.models.load_model(args.time_model)
    thr_model = tf.keras.models.load_model(args.thr_model)

    pred_time = float(time_model.predict(X, verbose=0).reshape(-1)[0])
    pred_thr = float(thr_model.predict(X, verbose=0).reshape(-1)[0])

    print("INPUT:")
    print("  depth:", args.depth)
    print("  base_filters:", args.base_filters)
    print("  batch_size:", args.batch_size)
    print("  params:", args.params)
    print("PREDICTION:")
    print("  predicted_avg_time_sec:", pred_time)
    print("  predicted_images_per_sec:", pred_thr)

if __name__ == "__main__":
    main()
