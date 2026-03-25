import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


# -------- 3. Training --------
def main():
    print("Loading SSL features...")
    df = pd.read_csv("./train_ssl_features.csv")

    # Select feature columns (all 'ssl_*')
    feature_cols = [c for c in df.columns if c.startswith("ssl_")]
    X = df[feature_cols].values
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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
        with torch.no_grad():
            for feats, labels in val_loader:
                logits = model(feats)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).float()
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss/len(val_loader):.4f} | "
            f"Val Acc: {100*correct/total:.2f}%"
        )

    torch.save(model.state_dict(), "ssl_mlp_weights.pt")
    print("Training complete. Weights saved to 'ssl_mlp_weights.pt'")


if __name__ == "__main__":
    main()
