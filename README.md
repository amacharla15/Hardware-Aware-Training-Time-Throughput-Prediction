# Hardware-Aware Training Time and Throughput Prediction for CNNs on A100

This project builds a **neural performance model** for deep learning workloads on an NVIDIA **A100** GPU.

Instead of predicting labels from images, the main goal is to **predict how long a training job will take** and how fast it will run:

> Given a CNN architecture and training hyperparameters,  
> **predict the time-per-epoch and images-per-second on an A100.**

The base workload is CIFAR-10 image classification with CNNs; on top of that, a dense ANN regressor is trained to learn the mapping from **model configuration → training performance**.

---

## Motivation

Modern ML teams don’t just care about accuracy – they care about:

- **Training time** (how long will this job run?)
- **Throughput** (images/sec, tokens/sec)
- **Cost and utilization** (how many GPU hours / dollars?)

Today, these questions are often answered by trial-and-error: “try this config and see how slow it is.”

This project is a small-scale prototype of an **empirical cost model** for CNN training on a GPU:

- Use CIFAR-10 CNNs as a realistic workload.
- Run many configurations on an A100 and **log real performance**.
- Train a neural regressor that can **predict runtime and throughput** for new CNN configurations, before actually running them.

This kind of approach is relevant to:

- Job-time and cost estimation on GPUs / accelerators (A100, H100, Trainium, Inferentia, etc.)
- Hyperparameter tuning under a **time budget**
- Cluster scheduling and capacity planning
- Comparing hardware (e.g., A100 vs other accelerators) with minimal extra runs

---

## High-Level Overview

The project has two ML components:

1. **Base model: CNNs on CIFAR-10**
   - Standard supervised image classification.
   - Vary:
     - `depth` (number of convolutional blocks)
     - `base_filters` (width)
     - `batch_size`
   - Train on an **A100** GPU.
   - Log:
     - `time_per_epoch` (seconds)
     - `images_per_second`
     - `num_params` (model size)
     - (optionally) `max_gpu_mem_mb` from `nvidia-smi`.

2. **Performance model: dense ANN regressor**
   - Input features (numeric):
     - `depth`
     - `base_filters`
     - `batch_size`
     - `num_params`
     - (optionally) dataset size, epochs, etc.
   - Targets:
     - `time_per_epoch`
     - (optionally) `images_per_second`
   - Trained as a regression model with MSE/MAE loss.
   - Evaluated with:
     - MAE / RMSE
     - R²
     - **Predicted vs actual** scatter plots.

The result is a model that can answer questions like:

> “If I train a 4-block CNN with 64 filters and batch size 128 on CIFAR-10,  
>  how long will each epoch take on an A100?”

---

## Repository Structure


```text
.
├─ notebooks/
│  ├─ 01_cifar10_cnn_baseline.ipynb       # Single CNN training on CIFAR-10 (sanity check)
│  ├─ 02_collect_cnn_perf_on_a100.ipynb   # Run config grid and log performance
│  └─ 03_train_performance_regressor.ipynb# Train/evaluate ANN regressor
├─ src/
│  ├─ models_cnn.py                       # CNN architecture factory
│  ├─ collect_metrics.py                  # Helpers for timing & logging
│  └─ perf_regressor.py                   # ANN regressor definition & training loop
├─ data/
│  ├─ configs_cifar_cnn_a100.csv          # Collected config ↔ performance dataset
│  └─ (optional) small_sample.csv         # Tiny sample for CPU-only demo
├─ plots/
│  ├─ accuracy_curves.png
│  ├─ batchsize_vs_images_per_sec.png
│  └─ predicted_vs_actual_time.png
├─ requirements.txt
└─ README.md
Dataset
Base dataset:

CIFAR-10 via tf.keras.datasets.cifar10

50,000 training images, 10 classes, 32×32 RGB

Performance dataset (generated):

Each row in configs_cifar_cnn_a100.csv corresponds to one training job run on the A100, with columns such as:

depth – number of conv blocks

base_filters – filters in the first block

batch_size

num_params

dataset_size

epochs

time_per_epoch

images_per_second

(optional) max_gpu_mem_mb

This CSV is the training data for the performance regressor.

Methods
1. CNN Architecture Family
A simple, scalable CNN template:

Repeatable conv “blocks”:

Conv2D(filters, 3×3) → ReLU → Conv2D(filters, 3×3) → ReLU → MaxPool(2×2)

Final layers:

Flatten → Dense(128) → ReLU → Dense(10, softmax)

Hyperparameters varied:

depth ∈ {2, 3, 4} (number of conv blocks)

base_filters ∈ {32, 64}

batch_size ∈ {32, 64, 128}

This yields a grid of configurations (e.g., 18 configs) to probe training performance.

2. Performance Logging on A100
For each configuration:

Build the CNN with given depth and base_filters.

Count parameters via model.count_params().

Train on CIFAR-10 for a small fixed number of epochs (e.g., 3).

Measure:

Wall-clock time per epoch (averaged)

Images per second = dataset_size / time_per_epoch

Optionally GPU memory from nvidia-smi.

Append a row to the CSV with config + metrics.

3. Performance Regressor
Features (X):

depth

base_filters

batch_size

num_params

(optional) dataset_size, epochs

Targets (y):

time_per_epoch (primary)

(optional) images_per_second

Model:

Small dense ANN (e.g., 2–3 hidden layers with ReLU)

Loss: MSE / MAE

Optimizer: Adam

Train/validation/test split on rows of configs_cifar_cnn_a100.csv.

Evaluation:

MAE, RMSE

R² on test set

Scatter plot: predicted vs actual time_per_epoch

How to Run
⚠️ Full experiments require access to an A100-like GPU 
A tiny CPU-only demo can be run by restricting configs and epochs.

1. Install dependencies
bash
Copy code
pip install -r requirements.txt
2. Run a single baseline CNN on CIFAR-10
bash
Copy code
# Inside notebooks/ or via Jupyter
# Open and run:
notebooks/01_cifar10_cnn_baseline.ipynb
This verifies TF/Keras + GPU are working and shows basic accuracy/loss curves.

3. Collect performance data on A100
bash
Copy code
# Open and run:
notebooks/02_collect_cnn_perf_on_a100.ipynb
This will:

Loop over the chosen hyperparameter grid,

Train each CNN for a few epochs,

Log config + time_per_epoch + images_per_second to data/configs_cifar_cnn_a100.csv.

4. Train the performance regressor
bash
Copy code
# Open and run:
notebooks/03_train_performance_regressor.ipynb
This will:

Load configs_cifar_cnn_a100.csv

Split into train/val/test

Train the ANN regressor

Print metrics and generate a predicted-vs-actual plot

Example Usage: What-If Prediction
Once the regressor is trained, you can query it with a hypothetical config:

python
Copy code
from src.perf_regressor import load_trained_model, predict_time

model = load_trained_model("checkpoints/perf_regressor.h5")

config = {
    "depth": 3,
    "base_filters": 64,
    "batch_size": 128,
    "num_params": 1_200_000,
}

pred_time = predict_time(model, config)
print(f"Predicted time per epoch on A100: {pred_time:.3f} s")
This enables “what-if” analysis:

Choose configs under a time budget (e.g., < 1.5 s/epoch)

Compare expected throughput across multiple candidate architectures

Results

Grid size: ~18 CNN configurations

Dataset: CIFAR-10 (50k training images, or a fixed subset)

Hardware: NVIDIA A100 (cscigpu), TensorFlow/Keras

Performance regressor:

R²: ~0.90 on held-out configs

MAE: ~0.05 s/epoch

Predicted-vs-actual scatter plot shows points clustered near the diagonal, indicating the model captures most of the variance in training time across configurations.

Real-World Use Cases
This small project is a prototype for:

Job-time & cost estimation
Estimate how long a training job will take on a given GPU before launching it.

Hyperparameter tuning under a time budget
Use the performance model as a filter: only try configs that fit within a specified epoch or wall-clock budget.

Cluster scheduling & planning
Extend the idea to many users/jobs so a scheduler can better predict queue times and choose where to place jobs.

Cross-hardware comparison (future work)
Repeat the same measurement procedure on other accelerators (e.g., AWS Trainium, Inferentia) and train separate or unified performance models to understand when each device is preferable.
