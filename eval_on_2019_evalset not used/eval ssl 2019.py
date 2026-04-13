import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import roc_curve

# --- 1. Re-declare the Lightweight SSL MLP Architecture ---
# This must exactly match the model you trained earlier
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

# --- 2. EER Calculation Function ---
def calculate_eer(y_true, y_scores):
    """Calculates Equal Error Rate using ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer * 100 # Return as percentage

def main():
    # 1. Load the Evaluation Data
    print("Loading ASVspoof 2019 LA Evaluation data (this might take a moment)...")
    eval_df = pd.read_csv("eval_la_ssl_features.csv")
    
    # Identify SSL feature columns
    feature_cols = [c for c in eval_df.columns if c.startswith("ssl_")]
    input_dim = len(feature_cols)
    
    # 2. Load the Scaler fitted on the 2019 LA training data
    print("Loading StandardScaler from training phase...")
    try:
        scaler = joblib.load("ssl_scaler.pkl")
    except FileNotFoundError:
        print("ERROR: Cannot find 'ssl_scaler.pkl'. Did you name it something else?")
        return

    # 3. Load the Trained SSL Model
    print(f"Loading trained SSL model with input dimension {input_dim}...")
    model = SSLMLP(input_dim)
    try:
        model.load_state_dict(torch.load("ssl_mlp_weights.pt"))
    except FileNotFoundError:
        print("ERROR: Cannot find 'ssl_mlp_weights.pt'.")
        return
        
    model.eval() # Set to evaluation mode
    
    # 4. Prepare Features
    X_eval = eval_df[feature_cols].values
    y_eval = eval_df['label'].values
    
    # Scale features
    X_eval_scaled = scaler.transform(X_eval)
    X_tensor = torch.tensor(X_eval_scaled, dtype=torch.float32)
    
    # 5. Batched Inference
    # We do this in batches so RAM doesn't crash on 71k files
    batch_size = 5000
    all_scores = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            logits = model(batch)
            
            # Convert logits to probabilities via sigmoid
            scores = torch.sigmoid(logits).numpy().flatten()
            all_scores.extend(scores)
            
    all_scores = np.array(all_scores)
    
    # 6. Calculate EER
    eer = calculate_eer(y_eval, all_scores)
    
    print(f"\n--- ASVspoof 2019 LA Evaluation (In-Domain) ---")
    print(f"Files tested: {len(eval_df)}")
    print(f"Equal Error Rate (EER): {eer:.3f}%")

if __name__ == "__main__":
    main()