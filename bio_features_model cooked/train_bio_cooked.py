import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve


# ---------------- 1. Reproducibility ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- 2. Dataset ----------------
class BioFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ---------------- 3. Model ----------------
class BioAcademicMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# ---------------- 4. Metrics ----------------
def calculate_eer(y_true, y_scores):
    y_true = np.asarray(y_true).reshape(-1)
    y_scores = np.asarray(y_scores).reshape(-1)

    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer * 100


# ---------------- 5. Feature prep ----------------
def prepare_features(df):
    df = df.copy()

    # Safety cleanup
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Feature engineering from the extracted academic set
    if "meanF0" in df.columns and "stdevF0" in df.columns:
        df["f0_cov"] = df["stdevF0"] / (df["meanF0"] + 1e-6)

    # Columns to log-transform
    log_cols = [
        "j_local", "j_abs", "j_rap", "j_ppq5", "j_ddp",
        "s_local", "s_db", "s_apq3", "s_apq5", "s_apq11", "s_dda"
    ]

    for col in log_cols:
        if col in df.columns:
            df[col] = np.log(np.clip(df[col].astype(float), 1e-6, None))

    feature_cols = [
        "meanF0", "f0_cov", "hnr",
        "j_local", "j_abs", "j_rap", "j_ppq5", "j_ddp",
        "s_local", "s_db", "s_apq3", "s_apq5", "s_apq11", "s_dda"
    ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    return X, y, feature_cols


# ---------------- 6. Training ----------------
def main():
    set_seed(42)

    csv_path = "./train_bio_features_academic.csv"
    scaler_path = "bio_academic_scaler.pkl"
    weights_path = "bio_academic_mlp_weights.pt"
    history_path = "bio_academic_training_history.csv"

    print("Loading extracted academic bio features...")
    df = pd.read_csv(csv_path)

    X, y, feature_cols = prepare_features(df)
    print(f"Loaded {len(df)} rows with {len(feature_cols)} final features.")

    # Split first, then fit scaler only on train
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to '{scaler_path}'")

    train_dataset = BioFeatureDataset(X_train, y_train)
    val_dataset = BioFeatureDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    input_dim = X_train.shape[1]
    model = BioAcademicMLP(input_dim)

    num_reals = (y_train == 1).sum()
    num_fakes = (y_train == 0).sum()
    pos_weight = torch.tensor([num_fakes / num_reals], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    epochs = 30
    patience = 7
    no_improve = 0
    best_val_eer = float("inf")
    history = []

    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        all_val_labels = []
        all_val_scores = []

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                logits = model(batch_features)
                loss = criterion(logits, batch_labels)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (logits > 0).float()

                total += batch_labels.size(0)
                correct += (preds == batch_labels).sum().item()

                all_val_labels.extend(batch_labels.cpu().numpy().flatten())
                all_val_scores.extend(probs.cpu().numpy().flatten())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_eer = calculate_eer(all_val_labels, all_val_scores)

        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "val_eer": val_eer,
            "lr": current_lr
        })

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Val EER: {val_eer:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        if val_eer < best_val_eer:
            best_val_eer = val_eer
            no_improve = 0
            torch.save(model.state_dict(), weights_path)
            print(f"  -> Validation EER improved to {val_eer:.2f}%. Saved '{weights_path}'")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"\nTraining history saved to '{history_path}'")
    print(f"Best weights saved to '{weights_path}' with Val EER = {best_val_eer:.2f}%")

if __name__ == "__main__":
    main()