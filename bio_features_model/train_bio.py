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
    # Setting fixed seeds ensures that neural network weights initialize the exact same way every time.
    # This prevents random variations in accuracy between test runs, proving the model is genuinely learning.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- 2. Dataset ----------------
class BioFeatureDataset(Dataset):
    def __init__(self, features, labels):
        # Converts standard Python arrays into PyTorch Tensors (the required format for GPU math).
        self.features = torch.tensor(features, dtype=torch.float32)
        # unsqueeze(1) changes the labels from a flat list [1, 0, 1] to a column vector [[1], [0], [1]].
        # This is strictly required for Binary Cross Entropy loss calculations.
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        # Allows PyTorch's DataLoader to know exactly how many rows of data exist to divide them into batches.
        return len(self.labels)

    def __getitem__(self, idx):
        # Fetches a specific row of features and its corresponding label when the DataLoader asks for it.
        return self.features[idx], self.labels[idx]


# ---------------- 3. Model ----------------
class BioAcademicMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # A lightweight Multi-Layer Perceptron (MLP) designed specifically for the 14 biological features.
        self.network = nn.Sequential(
            # The first hidden layer purposefully acts as a "bottleneck" (64 neurons) to force 
            # the model to learn general biological patterns rather than memorizing the exact training data.
            nn.Linear(input_dim, 64),
            # ReLU activation function introduces non-linearity so the network can learn complex patterns.
            nn.ReLU(),
            # BatchNorm standardizes the math passing between layers, keeping the network stable and learning fast.
            nn.BatchNorm1d(64),
            # Dropout randomly disables 30% of the neurons during training, acting as a strong defense against overfitting.
            nn.Dropout(0.3),

            # A second, smaller bottleneck layer (32 neurons).
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            # The final output layer compresses everything down into a single prediction (1 for real, 0 for fake).
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # This defines the "forward pass" — how the data flows through the sequential layers we built above.
        return self.network(x)


# ---------------- 4. Metrics ----------------
def calculate_eer(y_true, y_scores):
    # Ensures the lists are flattened into 1D arrays for Scikit-learn.
    y_true = np.asarray(y_true).reshape(-1)
    y_scores = np.asarray(y_scores).reshape(-1)

    # Calculates the False Positive Rate (fpr) and True Positive Rate (tpr) across various confidence thresholds.
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    # False Rejection Rate is simply the opposite of the True Positive Rate.
    fnr = 1 - tpr
    
    # EER is the exact point where the False Acceptance Rate equals the False Rejection Rate.
    # We find the index where the mathematical difference between the two arrays is the smallest.
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer * 100 # Returns as a neat percentage format for terminal printing


# ---------------- 5. Feature prep ----------------
def prepare_features(df):
    df = df.copy()

    # Safety cleanup: Praat occasionally returns infinity during division by zero errors. 
    # We convert those to NaN and drop them so they don't crash the neural network's loss functions.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Feature engineering: We calculate the 'Coefficient of Variation' for the fundamental frequency.
    # The 1e-6 acts as a mathematical safety net to guarantee we never divide by absolute zero.
    if "meanF0" in df.columns and "stdevF0" in df.columns:
        df["f0_cov"] = df["stdevF0"] / (df["meanF0"] + 1e-6)

    # Jitter and Shimmer values are extremely tiny but can have sharp exponential spikes.
    # We log-transform them to smooth out the data curve, making it much easier for the AI to learn.
    log_cols = [
        "j_local", "j_abs", "j_rap", "j_ppq5", "j_ddp",
        "s_local", "s_db", "s_apq3", "s_apq5", "s_apq11", "s_dda"
    ]

    for col in log_cols:
        if col in df.columns:
            # We clip the bottom value to 1e-6 because you cannot mathematically take the logarithm of zero or negative numbers.
            df[col] = np.log(np.clip(df[col].astype(float), 1e-6, None))

    # Define the exact 14 columns we expect to feed into the model.
    feature_cols = [
        "meanF0", "f0_cov", "hnr",
        "j_local", "j_abs", "j_rap", "j_ppq5", "j_ddp",
        "s_local", "s_db", "s_apq3", "s_apq5", "s_apq11", "s_dda"
    ]

    # Failsafe check to ensure Praat extracted every required biological feature properly.
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

    # Run the raw CSV data through the cleaning and feature engineering function we wrote above.
    X, y, feature_cols = prepare_features(df)
    print(f"Loaded {len(df)} rows with {len(feature_cols)} final features.")

    # We hold back 20% of the data (X_val) so the model can test itself on data it has never seen before.
    # stratify=y ensures both splits maintain the exact same ratio of real voices to deepfakes.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # StandardScaler normalizes the numbers so large numbers (pitch) don't overpower small numbers (jitter).
    scaler = StandardScaler()
    # CRITICAL: We only "fit" (calculate the math) on the training set, then strictly "transform" the validation set.
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # We save this specific mathematical scaler so we can perfectly replicate the scaling during the evaluation phase.
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to '{scaler_path}'")

    train_dataset = BioFeatureDataset(X_train, y_train)
    val_dataset = BioFeatureDataset(X_val, y_val)

    # We break the training data into manageable chunks of 64 files to optimize memory.
    # We shuffle the training data to prevent the model from learning patterns based on file order.
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    input_dim = X_train.shape[1]
    model = BioAcademicMLP(input_dim)

    # We calculate the ratio of fake audio files to real human audio files.
    num_reals = (y_train == 1).sum()
    num_fakes = (y_train == 0).sum()
    pos_weight = torch.tensor([num_fakes / num_reals], dtype=torch.float32)

    # We apply the ratio as a 'pos_weight'. This mathematically forces the AI to care more about 
    # wrongly flagging a real human as a deepfake, balancing the dataset out.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Adam is the optimizer that adjusts the model's weights. 
    # weight_decay (L2 Regularization) actively shrinks the weights towards zero, heavily preventing overfitting on the small feature set.
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # If the validation loss plateaus for 3 epochs, the scheduler cuts the learning rate in half so the AI can fine-tune its math.
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
        # Enable training features like Dropout
        model.train()
        train_loss = 0.0

        for batch_features, batch_labels in train_loader:
            # Wipe the old mathematical gradients away so they don't corrupt the new batch
            optimizer.zero_grad()
            # Forward pass: Generate predictions
            logits = model(batch_features)
            # Calculate how wrong the predictions were
            loss = criterion(logits, batch_labels)
            # Backward pass: Calculate the math required to fix the errors
            loss.backward()
            # Physically update the neural network weights
            optimizer.step()
            train_loss += loss.item()

        # Disable training features like Dropout because we are evaluating, not studying
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        all_val_labels = []
        all_val_scores = []

        # Turn off gradient tracking to save massive amounts of VRAM and speed up the evaluation
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                logits = model(batch_features)
                loss = criterion(logits, batch_labels)
                val_loss += loss.item()

                # Squashes the raw output numbers into neat 0.0 to 1.0 probability percentages
                probs = torch.sigmoid(logits)
                # If the probability is above 50%, lock in the guess as "Real" (1.0)
                preds = (logits > 0).float()

                total += batch_labels.size(0)
                correct += (preds == batch_labels).sum().item()

                # Save every exact percentage score to calculate the EER later
                all_val_labels.extend(batch_labels.cpu().numpy().flatten())
                all_val_scores.extend(probs.cpu().numpy().flatten())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_eer = calculate_eer(all_val_labels, all_val_scores)

        # Feed the validation loss into the scheduler to see if we need to slash the learning rate
        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        
        # Track the metrics to analyze the learning curve and plateau points later
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

        # If the Equal Error Rate hits a new record low, save the specific mathematical state of the network
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            no_improve = 0
            torch.save(model.state_dict(), weights_path)
            print(f"  -> Validation EER improved to {val_eer:.2f}%. Saved '{weights_path}'")
        else:
            no_improve += 1

        # If the model hasn't improved for 7 epochs, it is overfitting. Trigger the kill switch.
        if no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    # Save the tracked history to plot graphs later
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"\nTraining history saved to '{history_path}'")
    print(f"Best weights saved to '{weights_path}' with Val EER = {best_val_eer:.2f}%")

if __name__ == "__main__":
    main()