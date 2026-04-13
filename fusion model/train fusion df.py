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

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALER_PATH = os.path.join(BASE_DIR, "fusion_academic_scaler.pkl")
WEIGHTS_PATH = os.path.join(BASE_DIR, "fusion_academic_mlp_weights.pt")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class FusionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class FusionMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

def calculate_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return ((fpr[idx] + fnr[idx]) / 2.0) * 100

def main():
    set_seed(42)

    print("Loading SSL and Bio training features...")
    ssl_df = pd.read_csv(os.path.join(BASE_DIR, "train_ssl_features.csv"))
    bio_df = pd.read_csv(os.path.join(BASE_DIR, "train_bio_features_academic.csv"))
    
    bio_df = bio_df.drop(columns=["label"])
    
    print("Fusing datasets on matching filenames...")
    df = pd.merge(ssl_df, bio_df, on="filename", how="inner")
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    df["f0_cov"] = df["stdevF0"] / (df["meanF0"] + 1e-6)
    
    jsh_cols = [
        "j_local", "j_abs", "j_rap", "j_ppq5", "j_ddp",
        "s_local", "s_db", "s_apq3", "s_apq5", "s_apq11", "s_dda"
    ]
    for col in jsh_cols:
        df[col] = np.log(np.clip(df[col].astype(float), 1e-6, None))

    ssl_cols = [c for c in df.columns if c.startswith("ssl_")]
    bio_cols = ["meanF0", "f0_cov", "hnr"] + jsh_cols
    feature_cols = ssl_cols + bio_cols
    
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    print(f"Successfully fused! {len(df)} files | {len(feature_cols)} dimensions.")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved '{SCALER_PATH}'")

    train_loader = DataLoader(FusionDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(FusionDataset(X_val, y_val), batch_size=64, shuffle=False)

    model = FusionMLP(X_train.shape[1])

    num_reals = (y_train == 1).sum()
    num_fakes = (y_train == 0).sum()
    pos_weight = torch.tensor([num_fakes / num_reals], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    epochs = 30
    best_val_eer = float("inf")
    patience, no_improve = 7, 0

    print(f"\nStarting Fusion Model training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for feats, labels in train_loader:
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_val_labels, all_val_scores = [], []

        with torch.no_grad():
            for feats, labels in val_loader:
                logits = model(feats)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                preds = (logits > 0).float()
                
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                
                all_val_labels.extend(labels.cpu().numpy().flatten())
                all_val_scores.extend(probs.cpu().numpy().flatten())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_eer = calculate_eer(all_val_labels, all_val_scores)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val EER: {val_eer:.2f}%")

        if val_eer < best_val_eer:
            best_val_eer = val_eer
            no_improve = 0
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"  -> Validation EER improved! Saved weights.")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\nEarly stopping triggered.")
            break

    print(f"\nTraining complete. Best Fusion weights saved to '{WEIGHTS_PATH}'")

if __name__ == "__main__":
    main()