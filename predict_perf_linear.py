import argparse, math
import numpy as np
import joblib

def feat(depth, base_filters, batch_size, params):
    return np.array([[float(depth), float(base_filters), math.log2(float(batch_size)), math.log10(float(params))]], dtype=np.float32)

ap = argparse.ArgumentParser()
ap.add_argument("--depth", type=int, required=True)
ap.add_argument("--base_filters", type=int, required=True)
ap.add_argument("--batch_size", type=int, required=True)
ap.add_argument("--params", type=int, required=True)
args = ap.parse_args()

time_pipe = joblib.load("results/time_linear_pipeline.joblib")
thr_pipe  = joblib.load("results/throughput_linear_pipeline.joblib")

X = feat(args.depth, args.base_filters, args.batch_size, args.params)
pred_time = float(np.exp(time_pipe.predict(X)[0]))
pred_thr  = float(np.exp(thr_pipe.predict(X)[0]))


print("predicted_avg_time_sec:", pred_time)
print("predicted_images_per_sec:", pred_thr)
