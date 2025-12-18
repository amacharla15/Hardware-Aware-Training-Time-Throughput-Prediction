import csv, math
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def load_xy(target_key):
    X=[]
    y=[]
    with open("runs.csv", newline="") as f:
        for r in csv.DictReader(f):
            depth=float(r["depth"])
            base_filters=float(r["base_filters"])
            batch_size=float(r["batch_size"])
            params=float(r["params"])
            X.append([depth, base_filters, math.log2(batch_size), math.log10(params)])
            y.append(float(r[target_key]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_and_save(target_key, outpath):
    X,y = load_xy(target_key)
    y = np.log(y)  
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    pipe.fit(X,y)
    joblib.dump(pipe, outpath)
    print("saved", outpath)

train_and_save("avg_time_sec", "results/time_linear_pipeline.joblib")
train_and_save("images_per_sec", "results/throughput_linear_pipeline.joblib")
