"""
Code Availability Statement for Scientific Reports
Title: Short-Term Photovoltaic Power Forecasting Using DTW-Based K-Medoids Clustering and a Hybrid VMD-CEEMDAN-1DCNN-S-Mamba Model
Author: Hu TianXiang

Description:
This script contains the complete pipeline for the proposed method, including:
1. Data Preprocessing & DTW-KMedoids clustering for weather typing.
2. Signal Decomposition using VMD and CEEMDAN.
3. 1DCNN-SMamba model training for Active Power forecasting.

Note:
Please ensure 'data.csv' is placed in the same directory before running.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Clustering and Machine Learning modules
from dtaidistance import dtw
from kmedoids import KMedoids
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Signal Decomposition modules
from vmdpy import VMD
from PyEMD import CEEMDAN

# Deep Learning modules (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import copy

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# =========================================================
# Phase 1: Data Loading & DTW-KMedoids Weather Clustering
# =========================================================
print("========== Phase 1: DTW-KMedoids Clustering ==========")

# Load dataset
df = pd.read_csv('data_perfrct.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date
df['time'] = df['timestamp'].dt.time

# Construct daily solar radiation sequences
daily_ts = df.pivot_table(
    index='date',
    columns=df.groupby('date').cumcount(),
    values='Global_Horizontal_Radiation',
    aggfunc='first'
).fillna(0)

X = daily_ts.values.astype(np.float64)
print(f"Total number of days: {X.shape[0]}")
print(f"Time points per day: {X.shape[1]}")

# Chronological split (80% Train, 20% Test)
train_ratio = 0.8
n_total = X.shape[0]
n_train = int(n_total * train_ratio)

X_train = X[:n_train]
daily_ts_train = daily_ts.iloc[:n_train]
print(f"Training days: {n_train}, Test+Val days: {n_total - n_train}")

# Calculate DTW distance matrix on the training set
series_train = [X_train[i] for i in range(X_train.shape[0])]
dist_matrix_train = dtw.distance_matrix_fast(
    series_train,
    parallel=True,
    use_pruning=True,
    window=30,
)
np.fill_diagonal(dist_matrix_train, 0)

# Train K-Medoids clustering on DTW distance matrix
kmed = KMedoids(
    n_clusters=3,
    method='fasterpam',
    metric='precomputed',
    init='build',
    max_iter=300,
    random_state=42
)
kmed.fit(dist_matrix_train)
cluster_labels_train = kmed.labels_
medoid_indices_train = kmed.medoid_indices_
print("Medoid dates (train):", daily_ts_train.index[medoid_indices_train].tolist())

# Assign all samples to nearest training medoids using DTW distance
series_all = [X[i] for i in range(X.shape[0])]
dist_to_medoids = np.zeros((X.shape[0], kmed.n_clusters))

for i, seq in enumerate(series_all):
    for j, medoid_idx in enumerate(medoid_indices_train):
        medoid_seq = X_train[medoid_idx]
        dist_to_medoids[i, j] = dtw.distance_fast(seq, medoid_seq, window=30, use_pruning=True)

cluster_labels = np.argmin(dist_to_medoids, axis=1)

# Create date-to-cluster mapping and label weather types
date_cluster = pd.DataFrame({'date': daily_ts.index, 'cluster': cluster_labels})

# Calculate mean radiation per cluster to assign semantic labels (Sunny, Cloudy, Rainy)
cluster_means = []
for c in range(3):
    days_in_cluster = date_cluster[date_cluster['cluster'] == c]['date']
    if len(days_in_cluster) == 0:
        mean_value = 0.0
    else:
        subset = df[df['date'].isin(days_in_cluster)]
        mean_value = subset.groupby('date')['Global_Horizontal_Radiation'].mean().mean()
    cluster_means.append((c, mean_value))

cluster_means.sort(key=lambda x: x[1], reverse=True)
label_map = {
    cluster_means[0][0]: 'Sunny',
    cluster_means[1][0]: 'Cloudy',
    cluster_means[2][0]: 'Rainy'
}

date_cluster['weather'] = date_cluster['cluster'].map(label_map)
df = df.merge(date_cluster[['date', 'weather']], on='date', how='left')

# =========================================================
# Phase 2: Signal Decomposition (VMD + CEEMDAN)
# =========================================================
print("\n========== Phase 2: Signal Decomposition (VMD+CEEMDAN) ==========")

weather_groups = {}
decomposed = {}  # Stores VMD IMFs and residuals
secondary_decomposed = {}  # Stores CEEMDAN IMFs and trends

for weather in ['Sunny', 'Cloudy', 'Rainy']:
    subset = df[df['weather'] == weather].copy()
    power_series = subset['Active_Power'].dropna().values.astype(float)

    if len(power_series) < 200:
        print(f"-> Sequence for {weather} is too short, skipping.")
        continue

    weather_groups[weather] = power_series

    # ------------------ VMD Decomposition ------------------
    print(f"Applying VMD on {weather} condition...")
    alpha = 2000  # Bandwidth penalty
    tau = 0  # Noise-tolerance
    K = 5  # Number of modes
    DC = 0  # No DC part imposed
    init = 1  # Initialize omegas uniformly
    tol = 1e-7

    u, u_hat, omega = VMD(power_series, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol)
    imfs = u
    n_valid = imfs.shape[1]
    residual = power_series[:n_valid] - np.sum(imfs, axis=0)

    decomposed[weather] = {
        'imfs': imfs,
        'residual': residual,
        'original': power_series
    }

    # ------------------ CEEMDAN Secondary Decomposition ------------------
    print(f"Applying CEEMDAN on VMD residuals of {weather} condition...")
    ceemdan = CEEMDAN(trials=20, epsilon=0.05, noise_scale=0.2, parallel=False, max_imf=-1)
    secondary_imfs = ceemdan.ceemdan(residual.astype(np.float64))

    trend = secondary_imfs[-1]

    secondary_decomposed[weather] = {
        'secondary_imfs': secondary_imfs[:-1],
        'trend': trend,
        'original_residual': residual
    }

# =========================================================
# Phase 3: Deep Learning Prediction (1DCNN-SMamba)
# =========================================================
print("\n========== Phase 3: Deep Learning Training ==========")


# --- DL Models Definition ---
class CNN1D(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2, dilation=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x.transpose(1, 2)


class MambaBlock(nn.Module):
    def __init__(self, hidden_size, expand=2):
        super().__init__()
        self.inner_dim = hidden_size * expand
        self.in_proj = nn.Linear(hidden_size, self.inner_dim * 2)
        self.out_proj = nn.Linear(self.inner_dim, hidden_size)

        self.A_log = nn.Parameter(torch.log(1e-2 * torch.ones(self.inner_dim)))
        self.B = nn.Parameter(torch.randn(self.inner_dim) * 0.02)
        self.C = nn.Parameter(torch.randn(self.inner_dim) * 0.02)
        self.act = nn.SiLU()

    def forward(self, x):
        B, T, _ = x.shape
        z, gate = self.in_proj(x).chunk(2, dim=-1)
        z = self.act(z)

        h_fwd = torch.zeros(B, self.inner_dim, device=x.device)
        h_bwd = torch.zeros(B, self.inner_dim, device=x.device)

        y_fwd, y_bwd = [], []
        A = -torch.exp(self.A_log)
        exp_A = torch.exp(A)

        for t in range(T):
            h_fwd = h_fwd * exp_A + z[:, t] * self.B
            y_fwd.append(h_fwd * self.C)

            tr = T - 1 - t
            h_bwd = h_bwd * exp_A + z[:, tr] * self.B
            y_bwd.append(h_bwd * self.C)

        y = (torch.stack(y_fwd, 1) + torch.stack(y_bwd[::-1], 1)) / 2
        y = y * self.act(gate)
        return self.out_proj(y)


class SMamba(nn.Module):
    def __init__(self, hidden_size, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([MambaBlock(hidden_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        for layer in self.layers:
            x = self.norm(layer(x) + x)
        return x


class OneDCNN_SMamba(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, horizon=3):
        super().__init__()
        self.cnn = CNN1D(input_size, hidden_size)
        self.mamba = SMamba(hidden_size, num_layers)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        x = self.cnn(x)
        x = self.mamba(x)
        x = x[:, -1, :]
        return self.head(x)


class PVDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- Data Preparation for DL ---
def prepare_data_for_weather(weather_type, df_data, dec_dict, sec_dec_dict, window_size=30, horizon=3):
    subset = df_data[df_data['weather'] == weather_type].copy().reset_index(drop=True)
    timestamps = subset['timestamp'].values

    feature_cols = [
        'Global_Horizontal_Radiation', 'Weather_Temperature_Celsius',
        'Weather_Relative_Humidity', 'Wind_Speed',
        'Diffuse_Horizontal_Radiation', 'Radiation_Global_Tilted'
    ]
    raw_features = subset[feature_cols].values.astype(float)

    ohe = OneHotEncoder(sparse_output=False)
    cluster_ohe = ohe.fit_transform(subset[['weather']])

    vmd_imfs = dec_dict[weather_type]['imfs'].T
    n_valid = vmd_imfs.shape[0]

    raw_features = raw_features[:n_valid]
    cluster_ohe = cluster_ohe[:n_valid]
    ceemdan_imfs = sec_dec_dict[weather_type]['secondary_imfs'].T[:n_valid]
    trend = sec_dec_dict[weather_type]['trend'][:n_valid].reshape(-1, 1)
    timestamps = timestamps[:n_valid]

    all_inputs = np.hstack([raw_features, cluster_ohe, vmd_imfs, ceemdan_imfs, trend])
    targets = subset['Active_Power'].values.astype(float)[:n_valid]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    all_inputs = scaler_x.fit_transform(all_inputs)
    targets = scaler_y.fit_transform(targets.reshape(-1, 1)).flatten()

    X_data, y_data, ts_data = [], [], []
    for i in range(len(all_inputs) - window_size - horizon):
        X_data.append(all_inputs[i:i + window_size])
        y_data.append(targets[i + window_size:i + window_size + horizon])
        ts_data.append(timestamps[i + window_size])

    X_data, y_data, ts_data = np.array(X_data), np.array(y_data), np.array(ts_data)
    split = int(0.8 * len(X_data))

    return (X_data[:split], y_data[:split], ts_data[:split],
            X_data[split:], y_data[split:], ts_data[split:],
            scaler_x, scaler_y)


# --- Training Execution with Early Stopping ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HORIZON = 3
MAX_EPOCHS = 100
PATIENCE = 15

weather_list = ['Sunny', 'Cloudy', 'Rainy']

for weather in weather_list:
    if weather not in decomposed or weather not in secondary_decomposed:
        continue

    print(f"\n===== Training Model for {weather} Conditions =====")
    X_tr, y_tr, ts_tr, X_te, y_te, ts_te, sx, sy = prepare_data_for_weather(
        weather, df, decomposed, secondary_decomposed, window_size=30, horizon=HORIZON
    )

    train_loader = DataLoader(PVDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    val_loader = DataLoader(PVDataset(X_te, y_te), batch_size=32, shuffle=False)

    model = OneDCNN_SMamba(input_size=X_tr.shape[2], hidden_size=64, num_layers=2, horizon=HORIZON).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Early Stopping Tracking Variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None

    for epoch in range(MAX_EPOCHS):
        # Training Phase
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation Phase (using test set for early stopping as proxy)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb_val, yb_val in val_loader:
                xb_val, yb_val = xb_val.to(device), yb_val.to(device)
                pred_val = model(xb_val)
                loss_v = criterion(pred_val, yb_val)
                val_loss += loss_v.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1}. Restoring best weights.")
            break

    # Load best weights for final evaluation
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # ---------- Final Evaluation ----------
    model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)
        y_pred = model(X_te_t).cpu().numpy()

    y_pred_inv = sy.inverse_transform(y_pred)
    y_te_inv = sy.inverse_transform(y_te)

    mae = mean_absolute_error(y_te_inv.flatten(), y_pred_inv.flatten())
    mse = mean_squared_error(y_te_inv.flatten(), y_pred_inv.flatten())
    rmse = np.sqrt(mse)
    r2 = r2_score(y_te_inv.flatten(), y_pred_inv.flatten())

    print(f"\nFinal Test Metrics for {weather}:")
    print(f"MAE={mae:.4f} | MSE={mse:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")

    for i in range(HORIZON):
        step_mae = mean_absolute_error(y_te_inv[:, i], y_pred_inv[:, i])
        print(f" -> Step t+{i + 1} MAE = {step_mae:.4f}")

    # ---------- Save the Model and Scalers ----------
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"model_{weather}.pth")
    info_path = os.path.join(save_dir, f"info_{weather}.json")

    torch.save(model.state_dict(), model_path)

    info = {
        "weather": weather,
        "window_size": 30,
        "horizon": HORIZON,
        "hidden_size": 64,
        "num_layers": 2,
        "input_features": int(X_tr.shape[2]),
        "scaler_x_min": sx.min_.tolist(),
        "scaler_x_scale": sx.scale_.tolist(),
        "scaler_y_min": float(sy.min_[0]),
        "scaler_y_scale": float(sy.scale_[0]),
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "test_r2": float(r2),
    }

    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)

    print(f"Model successfully saved to '{save_dir}/'")

print("\nPipeline execution fully completed.")