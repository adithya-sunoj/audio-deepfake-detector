import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import roc_curve

# -------- 1. Re-declare the Lightweight MLP Architecture --------
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

# -------- 2. EER Calculation Function --------
def calculate_eer(y_true, y_scores):
    """Calculates Equal Error Rate using ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    frr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute(fpr - frr))]
    return eer * 100 # return as percentage

# -------- 3. Main Evaluation --------
def main():
    print("Loading evaluation data (this might take a moment)...")
    # Load the ~600k file CSV you just generated
    eval_df = pd.read_csv("eval_df_ssl_features.csv")
    
    # Identify the feature columns dynamically
    feature_cols = [c for c in eval_df.columns if c.startswith("ssl_")]
    input_dim = len(feature_cols)
    
    # 1. Load the Scaler fitted on the training data
    print("Loading StandardScaler from training phase...")
    try:
        scaler = joblib.load("ssl_scaler.pkl")
    except FileNotFoundError:
        print("ERROR: Cannot find 'ssl_scaler.pkl'. Please ensure it is in the same directory.")
        return

    # 2. Load the Model
    print(f"Loading trained model with input dimension {input_dim}...")
    model = SSLMLP(input_dim)
    try:
        model.load_state_dict(torch.load("ssl_mlp_weights.pt"))
    except FileNotFoundError:
        print("ERROR: Cannot find 'ssl_mlp_weights.pt'. Please ensure it is in the same directory.")
        return
        
    model.eval() # Set to evaluation mode
    
    # 3. Separate into Clean (nocodec) and Compressed (everything else)
    clean_df = eval_df[eval_df['codec'] == 'nocodec']
    compressed_df = eval_df[eval_df['codec'] != 'nocodec']
    
    datasets = {"Clean (No Codec)": clean_df, "Compressed": compressed_df}
    
    for condition, df in datasets.items():
        if len(df) == 0:
            print(f"\n--- Condition: {condition} Audio ---")
            print("No files found for this condition.")
            continue
            
        X_eval = df[feature_cols].values
        y_eval = df['label'].values
        
        # Scale the features using the loaded scaler
        X_eval_scaled = scaler.transform(X_eval)
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_eval_scaled, dtype=torch.float32)
        
        # 4. Get Predictions
        # We do this in batches manually so we don't blow up the RAM trying to process 600k files at once
        batch_size = 5000
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                logits = model(batch)
                scores = torch.sigmoid(logits).numpy().flatten()
                all_scores.extend(scores)
                
        all_scores = np.array(all_scores)
        
        # Calculate and print EER
        eer = calculate_eer(y_eval, all_scores)
        print(f"\n--- Condition: {condition} Audio ---")
        print(f"Files tested: {len(df)}")
        print(f"Equal Error Rate (EER): {eer:.2f}%")

if __name__ == "__main__":
    main()