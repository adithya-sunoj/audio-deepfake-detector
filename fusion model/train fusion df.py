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
# Dynamically resolves the root directory of the project to avoid hardcoded paths breaking on different machines
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Paths for the persistent artifacts that must be identical during the evaluation phase
SCALER_PATH = os.path.join(BASE_DIR, "fusion_academic_scaler.pkl")
WEIGHTS_PATH = os.path.join(BASE_DIR, "fusion_academic_mlp_weights.pt")

def set_seed(seed=42):
    """
    Ensures absolute mathematical reproducibility across runs. 
    Fixes the random number generators for standard Python, NumPy, and PyTorch (both CPU and GPU).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class FusionDataset(Dataset):
    """
    A custom PyTorch Dataset wrapper. 
    Required for the DataLoader to efficiently stream batches of data to the GPU.
    """
    def __init__(self, features, labels):
        # Convert raw numpy arrays into PyTorch float32 tensors. 
        self.features = torch.tensor(features, dtype=torch.float32)
        # unsqueeze(1) converts a flat array like [1, 0, 1] into a column vector [[1], [0], [1]].
        # This exact shape is required by PyTorch's BCEWithLogitsLoss function.
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        # Tells the DataLoader the total number of samples in the dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # Returns a single matched pair of (features, label) for a given index
        return self.features[idx], self.labels[idx]

class FusionMLP(nn.Module):
    """
    A Multi-Layer Perceptron designed to fuse and classify high-dimensional SSL embeddings 
    and low-dimensional biological acoustic features.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            # First hidden layer: compresses the large feature space
            nn.Linear(input_dim, 512),
            nn.ReLU(), # Adds non-linearity to learn complex patterns
            nn.BatchNorm1d(512), # Normalizes layer outputs to stabilize and speed up training
            nn.Dropout(0.4), # Randomly disables 40% of neurons during training to prevent memorization/overfitting

            # Second hidden layer: further compression
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            # Final bottleneck layer before output
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Output layer: condenses down to a single binary logit (Real vs Fake)
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Defines the forward pass through the network
        return self.network(x)

def calculate_eer(y_true, y_scores):
    """
    Calculates the Equal Error Rate (EER) - the standard security metric where 
    False Acceptance Rate equals False Rejection Rate.
    """
    # Computes the ROC curve across varying confidence thresholds
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr # False Negative Rate
    # Finds the index where the absolute difference between False Positives and False Negatives is closest to zero
    idx = np.nanargmin(np.abs(fpr - fnr))
    # Averages them at the crossing point and returns as a percentage
    return ((fpr[idx] + fnr[idx]) / 2.0) * 100

def main():
    set_seed(42)

    print("Loading SSL and Bio training features...")
    ssl_df = pd.read_csv(os.path.join(BASE_DIR, "train_ssl_features.csv"))
    bio_df = pd.read_csv(os.path.join(BASE_DIR, "train_bio_features_academic.csv"))
    
    # Drop the duplicate label column to prevent conflicts during the merge
    bio_df = bio_df.drop(columns=["label"])
    
    print("Fusing datasets on matching filenames...")
    # 'inner' join ensures that if a file failed Praat extraction and is missing from bio_df, 
    # it is automatically dropped from the final dataset to maintain exact row alignment
    df = pd.merge(ssl_df, bio_df, on="filename", how="inner")
    
    # Data safety: removes mathematically impossible values generated during extraction
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Feature Engineering: Coefficient of Variation normalizes pitch variation relative to base pitch.
    # The +1e-6 prevents division by zero if an audio file was entirely monotone.
    df["f0_cov"] = df["stdevF0"] / (df["meanF0"] + 1e-6)
    
    jsh_cols = [
        "j_local", "j_abs", "j_rap", "j_ppq5", "j_ddp",
        "s_local", "s_db", "s_apq3", "s_apq5", "s_apq11", "s_dda"
    ]
    # Logarithmic transformation for Jitter/Shimmer features. 
    # These values are tiny but have massive exponential outliers. 
    # Log-scaling smooths the distribution so the Neural Network gradients don't explode.
    for col in jsh_cols:
        # np.clip acts as a floor, preventing log(0) errors.
        df[col] = np.log(np.clip(df[col].astype(float), 1e-6, None))

    # Dynamically grab the hundreds of Wav2Vec2 columns and combine them with the bio columns
    ssl_cols = [c for c in df.columns if c.startswith("ssl_")]
    bio_cols = ["meanF0", "f0_cov", "hnr"] + jsh_cols
    feature_cols = ssl_cols + bio_cols
    
    # Isolate the final features and labels as raw float32 NumPy arrays for PyTorch
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    print(f"Successfully fused! {len(df)} files | {len(feature_cols)} dimensions.")

    # Stratified split ensures the 80/20 train/val ratio has the exact same proportion of deepfakes to real voices
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardization prevents large numbers from overpowering small numbers during gradient descent
    scaler = StandardScaler()
    # Fit the scaler ONLY on the training data to prevent data leakage into the validation set
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Save the scaler. The exact same mathematical bounds must be used during the massive evaluation phase later.
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved '{SCALER_PATH}'")

    # Wrap the scaled data in DataLoaders for dynamic batch streaming
    train_loader = DataLoader(FusionDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(FusionDataset(X_val, y_val), batch_size=64, shuffle=False)

    # Initialize the architecture with the dynamic input dimension (approx. 1550 features)
    model = FusionMLP(X_train.shape[1])

    # Calculate class weights. Deepfake datasets are heavily imbalanced (e.g., 10x more fakes than reals).
    # This weight heavily penalizes the model for misclassifying a rare "real" voice.
    num_reals = (y_train == 1).sum()
    num_fakes = (y_train == 0).sum()
    pos_weight = torch.tensor([num_fakes / num_reals], dtype=torch.float32)

    # Binary Cross Entropy loss, initialized with the class imbalance weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Adam optimizer with L2 Regularization (weight_decay) to actively shrink weights and prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Dynamic learning rate. If validation loss plateaus for 3 epochs, cut the learning rate in half to fine-tune
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    epochs = 30
    best_val_eer = float("inf")
    # Early stopping config: halt script if EER doesn't improve for 7 epochs
    patience, no_improve = 7, 0

    print(f"\nStarting Fusion Model training for {epochs} epochs...")

    for epoch in range(epochs):
        # Enable training features (Dropout, BatchNorm tracking)
        model.train()
        train_loss = 0.0
        
        # Training loop iteration
        for feats, labels in train_loader:
            optimizer.zero_grad() # Clear old gradients
            logits = model(feats) # Forward pass
            loss = criterion(logits, labels) # Calculate error
            loss.backward() # Backpropagation to compute gradient updates
            optimizer.step() # Apply the updates to the weights
            train_loss += loss.item()

        # Switch to evaluation mode (disables Dropout for consistent testing)
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_val_labels, all_val_scores = [], []

        # Disable gradient tracking to save VRAM and speed up inference
        with torch.no_grad():
            for feats, labels in val_loader:
                logits = model(feats)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                # Squash raw logits into 0.0-1.0 probabilities for accuracy and EER metrics
                probs = torch.sigmoid(logits)
                preds = (logits > 0).float() # Standard 0.5 threshold for accuracy metric
                
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                
                # Append batch arrays to master lists for whole-epoch EER calculation
                all_val_labels.extend(labels.cpu().numpy().flatten())
                all_val_scores.extend(probs.cpu().numpy().flatten())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_eer = calculate_eer(all_val_labels, all_val_scores)
        
        # Step the scheduler using the validation loss
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val EER: {val_eer:.2f}%")

        # Core Checkpointing Logic: Only save the model to disk if it achieved a new record-low EER
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            no_improve = 0
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"  -> Validation EER improved! Saved weights.")
        else:
            no_improve += 1

        # Early Stopping Kill-switch
        if no_improve >= patience:
            print(f"\nEarly stopping triggered.")
            break

    print(f"\nTraining complete. Best Fusion weights saved to '{WEIGHTS_PATH}'")

if __name__ == "__main__":
    main()