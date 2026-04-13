import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import roc_curve

# --- 1. The Academic MLP Architecture ---
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

# --- 2. Function to Calculate EER ---
def calculate_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer * 100

# --- 3. Feature Prep (Must perfectly match training) ---
def prepare_features(df):
    df = df.copy()
    
    # Safety cleanup
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Feature engineering
    if "meanF0" in df.columns and "stdevF0" in df.columns:
        df["f0_cov"] = df["stdevF0"] / (df["meanF0"] + 1e-6)

    # Log transforms
    jsh_cols = [
        "j_local", "j_abs", "j_rap", "j_ppq5", "j_ddp",
        "s_local", "s_db", "s_apq3", "s_apq5", "s_apq11", "s_dda"
    ]
    for col in jsh_cols:
        if col in df.columns:
            df[col] = np.log(np.clip(df[col].astype(float), 1e-6, None))

    feature_cols = ["meanF0", "f0_cov", "hnr"] + jsh_cols
    
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    
    return X, y

def main():
    # 1. Load the Scaler
    print("Loading scaler from training phase...")
    scaler = joblib.load("bio_academic_scaler.pkl")

    # 2. Load the Model
    input_dim = 14
    model = BioAcademicMLP(input_dim)
    model.load_state_dict(torch.load("bio_academic_mlp_weights.pt"))
    model.eval()
    
    # 3. Load the Evaluation Data
    print("Loading evaluation dataset (this may take a moment)...")
    eval_df = pd.read_csv("eval_df_bio_academic.csv")
    
    # Separate into Clean (nocodec) and Compressed
    clean_df = eval_df[eval_df['codec'] == 'nocodec']
    compressed_df = eval_df[eval_df['codec'] != 'nocodec']
    
    datasets = {"Clean (No Codec)": clean_df, "Compressed": compressed_df}
    
    for condition, df in datasets.items():
        if len(df) == 0:
            print(f"\n--- Condition: {condition} Audio ---")
            print("No files found for this condition.")
            continue
            
        # Prep features (f0_cov, log transform, etc.)
        X_eval, y_eval = prepare_features(df)
        
        # Scale the features
        X_eval_scaled = scaler.transform(X_eval)
        X_tensor = torch.tensor(X_eval_scaled, dtype=torch.float32)
        
        # 4. Batched Inference (To prevent RAM exhaustion)
        batch_size = 5000
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                logits = model(batch)
                scores = torch.sigmoid(logits).numpy().flatten()
                all_scores.extend(scores)
                
        # Calculate and print EER
        eer = calculate_eer(y_eval, all_scores)
        print(f"\n--- Condition: {condition} Audio ---")
        print(f"Files tested: {len(X_eval)}")  # using X_eval len as some might have dropped
        print(f"Equal Error Rate (EER): {eer:.2f}%")

if __name__ == "__main__":
    main()