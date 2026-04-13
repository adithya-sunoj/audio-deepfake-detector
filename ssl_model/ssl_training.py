import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve  # <-- Added for EER


# -------- 1. Dataset --------
class SSLFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# -------- 2. Lightweight MLP --------
class SSLMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x)


# -------- EER Calculation Function --------
def calculate_eer(y_true, y_scores):
    """Calculates Equal Error Rate using ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    frr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute(fpr - frr))]
    return eer * 100 # return as percentage


# -------- 3. Training --------
def main():
    print("Loading SSL features...")
    df = pd.read_csv("./train_ssl_features.csv")

    feature_cols = [c for c in df.columns if c.startswith("ssl_")]
    X = df[feature_cols].values
    y = df["label"].values

    print(f"Loaded {len(df)} files with {X.shape[1]} dimensions.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, 'ssl_scaler.pkl')
    print("Saved 'ssl_scaler.pkl'")

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = SSLFeatureDataset(X_train, y_train)
    val_dataset = SSLFeatureDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    input_dim = X_train.shape[1]
    model = SSLMLP(input_dim)

    num_reals = (y_train == 1).sum()
    num_fakes = (y_train == 0).sum()
    pos_weight = torch.tensor([num_fakes / num_reals], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    best_val_eer = float('inf') # <-- Track best EER instead of loss

    print(f"Starting training for {epochs} epochs...")

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
        val_loss = 0.0
        correct = 0
        total = 0
        
        # Arrays to collect all labels and scores for EER calculation
        all_val_labels = []
        all_val_scores = []
        
        with torch.no_grad():
            for feats, labels in val_loader:
                logits = model(feats)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                scores = torch.sigmoid(logits)
                preds = (scores > 0.5).float()
                
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                
                # Store for EER calculation
                all_val_labels.extend(labels.numpy())
                all_val_scores.extend(scores.numpy())

        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        val_acc = 100*correct/total
        
        # Calculate EER
        val_eer = calculate_eer(np.array(all_val_labels), np.array(all_val_scores))

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Val EER: {val_eer:.2f}%"  # <-- Print EER
        )

        # Save checkpoint based on best EER
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            torch.save(model.state_dict(), "ssl_mlp_weights.pt")
            print(f"  -> Validation EER improved to {val_eer:.2f}%! Saved 'ssl_mlp_weights.pt'")

    print(f"\nTraining complete. Best weights (EER: {best_val_eer:.2f}%) saved to 'ssl_mlp_weights.pt'")

if __name__ == "__main__":
    main()