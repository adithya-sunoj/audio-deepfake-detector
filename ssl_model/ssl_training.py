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
    """
    Custom PyTorch Dataset to handle the Wav2Vec2 (SSL) embeddings.
    Converts raw numerical arrays into PyTorch tensors for GPU processing.
    """
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        # unsqueeze(1) converts a flat array like [1, 0, 1] into a column matrix [[1], [0], [1]].
        # This specific shape is strictly required by PyTorch's binary classification loss functions.
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Allows PyTorch's DataLoader to fetch specific rows efficiently during batch generation
        return self.features[idx], self.labels[idx]


# -------- 2. Lightweight MLP --------
class SSLMLP(nn.Module):
    """
    A Multi-Layer Perceptron designed to classify the high-dimensional Wav2Vec2 embeddings.
    Kept intentionally lightweight to prevent overfitting and run efficiently on constrained hardware.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            # First hidden layer compresses the large SSL embedding down to 256 dimensions
            nn.Linear(input_dim, 256),
            nn.ReLU(),  # Introduces non-linearity so the network can learn complex patterns
            
            # Batch Normalization standardizes the outputs of the previous layer. 
            # This prevents the mathematical gradients from collapsing and speeds up training.
            nn.BatchNorm1d(256),
            
            # Dropout randomly turns off 30% of the neurons during each training step.
            # This forces the network to generalize rather than simply memorizing the training data.
            nn.Dropout(0.3),

            # Second hidden layer compresses further to 128 dimensions
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Final output layer compresses down to exactly 1 number (the real/fake prediction)
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x)


# -------- EER Calculation Function --------
def calculate_eer(y_true, y_scores):
    """
    Calculates the Equal Error Rate (EER), the gold-standard security metric for biometrics.
    EER is the exact threshold where the False Acceptance Rate equals the False Rejection Rate.
    """
    # roc_curve evaluates performance across hundreds of different confidence thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    
    # False Rejection Rate is the inverse of the True Positive Rate
    frr = 1 - tpr
    
    # Find the exact index where the gap between FPR and FRR is the absolute smallest
    eer = fpr[np.nanargmin(np.absolute(fpr - frr))]
    return eer * 100 # Return as a clean percentage


# -------- 3. Training --------
def main():
    print("Loading SSL features...")
    df = pd.read_csv("./train_ssl_features.csv")

    # Dynamically extract all Wav2Vec2 feature columns (ignoring metadata columns like filename)
    feature_cols = [c for c in df.columns if c.startswith("ssl_")]
    X = df[feature_cols].values
    y = df["label"].values

    print(f"Loaded {len(df)} files with {X.shape[1]} dimensions.")

    # Neural networks struggle if data variance is too large. 
    # StandardScaler normalizes all features to have a mean of 0 and a standard deviation of 1.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # CRITICAL: We must save this fitted scaler to disk. Any future audio files evaluated 
    # by this model must be scaled using this exact same mathematical reference point.
    joblib.dump(scaler, 'ssl_scaler.pkl')
    print("Saved 'ssl_scaler.pkl'")

    # Split data 80/20. 'stratify=y' guarantees the ratio of real vs fake audio is 
    # perfectly identical in both the training and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = SSLFeatureDataset(X_train, y_train)
    val_dataset = SSLFeatureDataset(X_val, y_val)

    # DataLoaders group the data into batches of 64 so the GPU doesn't run out of memory.
    # We shuffle the training data to prevent the model from learning repeating sequence patterns.
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    input_dim = X_train.shape[1]
    model = SSLMLP(input_dim)

    # Deepfake datasets have severe class imbalance (often 10x more fakes than reals).
    # We calculate the imbalance ratio to penalize the model mathematically if it just guesses "fake" every time.
    num_reals = (y_train == 1).sum()
    num_fakes = (y_train == 0).sum()
    pos_weight = torch.tensor([num_fakes / num_reals], dtype=torch.float32)

    # BCEWithLogitsLoss is mathematically more stable than running a Sigmoid followed by standard BCELoss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # The Adam optimizer updates the network's weights based on the loss gradients
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    # We track the best EER (lowest is better) instead of standard loss, as EER represents true system robustness
    best_val_eer = float('inf') 

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # 1. TRAINING PHASE
        model.train() # Activates training-specific layers like Dropout and BatchNorm
        train_loss = 0.0
        
        for feats, labels in train_loader:
            optimizer.zero_grad() # Wipes the mathematical gradients from the previous batch
            
            logits = model(feats) # Forward pass: Model makes its predictions
            loss = criterion(logits, labels) # Calculate how wrong the predictions were
            
            loss.backward() # Backward pass: Calculate the gradients needed to fix the error
            optimizer.step() # Apply the gradients to update the model's internal weights
            
            train_loss += loss.item()

        # 2. VALIDATION PHASE
        model.eval() # Deactivates Dropout/BatchNorm so the model performs purely deterministic inference
        val_loss = 0.0
        correct = 0
        total = 0
        
        # Arrays to collect all labels and precise probability scores for the EER calculation
        all_val_labels = []
        all_val_scores = []
        
        # torch.no_grad() disables gradient tracking, massively reducing RAM usage and speeding up inference
        with torch.no_grad():
            for feats, labels in val_loader:
                logits = model(feats)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                # Sigmoid squashes the raw output logit into a clean 0.0 to 1.0 probability percentage
                scores = torch.sigmoid(logits)
                
                # If the probability is > 50%, lock in the prediction as a 1 (Real). Otherwise, 0 (Fake).
                preds = (scores > 0.5).float()
                
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                
                # Store the exact scores and labels for the EER curve
                all_val_labels.extend(labels.numpy())
                all_val_scores.extend(scores.numpy())

        # 3. METRICS AND CHECKPOINTING
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        val_acc = 100*correct/total
        
        # Calculate the Equal Error Rate using the collected predictions
        val_eer = calculate_eer(np.array(all_val_labels), np.array(all_val_scores))

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Val EER: {val_eer:.2f}%"  # <-- Print EER
        )

        # Checkpoint logic: Only save the model to the hard drive if the EER hits a new record low.
        # This ensures we always keep the most generalized version of the model, even if it overfits later.
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            torch.save(model.state_dict(), "ssl_mlp_weights.pt")
            print(f"  -> Validation EER improved to {val_eer:.2f}%! Saved 'ssl_mlp_weights.pt'")

    print(f"\nTraining complete. Best weights (EER: {best_val_eer:.2f}%) saved to 'ssl_mlp_weights.pt'")

if __name__ == "__main__":
    main()