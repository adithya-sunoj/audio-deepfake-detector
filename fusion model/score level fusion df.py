import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve
from tqdm import tqdm

# ==========================================
# 1. FILE PATHS (UPDATE THESE IF NEEDED)
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data
SSL_CSV = os.path.join(BASE_DIR, "eval_df_ssl_features.csv")
BIO_CSV = os.path.join(BASE_DIR, "eval_df_bio_academic.csv")

# SSL Standalone Model
SSL_SCALER_PATH = os.path.join(BASE_DIR, "ssl_scaler.pkl")
SSL_WEIGHTS_PATH = os.path.join(BASE_DIR, "ssl_mlp_weights.pt")

# Bio Standalone Model
BIO_SCALER_PATH = os.path.join(BASE_DIR, "bio_academic_scaler.pkl")
BIO_WEIGHTS_PATH = os.path.join(BASE_DIR, "bio_academic_mlp_weights.pt")

# ==========================================
# 2. NEURAL NETWORK ARCHITECTURES
# ==========================================
class SSLModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),         # network.0
            nn.ReLU(),                         # network.1
            nn.BatchNorm1d(256),               # network.2
            nn.Dropout(0.4),                   # network.3
            nn.Linear(256, 128),               # network.4
            nn.ReLU(),                         # network.5
            nn.Dropout(0.3),                   # network.6 (No BatchNorm here!)
            nn.Linear(128, 1)                  # network.7
        )
    def forward(self, x):
        return self.network(x)

class BioModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),          # network.0
            nn.ReLU(),                         # network.1
            nn.BatchNorm1d(64),                # network.2
            nn.Dropout(0.4),                   # network.3
            nn.Linear(64, 32),                 # network.4
            nn.ReLU(),                         # network.5
            nn.Dropout(0.3),                   # network.6
            nn.Linear(32, 1)                   # network.7
        )
    def forward(self, x):
        return self.network(x)
    
def calculate_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return ((fpr[idx] + fnr[idx]) / 2.0) * 100

# ==========================================
# 3. RAM-SAFE MULTI-MODEL DATASET
# ==========================================
class MultiModelEvalDataset(Dataset):
    def __init__(self, dataframe, ssl_scaler, bio_scaler, ssl_cols, bio_cols):
        self.df = dataframe
        self.ssl_scaler = ssl_scaler
        self.bio_scaler = bio_scaler
        self.ssl_cols = ssl_cols
        self.bio_cols = bio_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Extract and scale SSL features for this row
        ssl_features = row[self.ssl_cols].values.astype(np.float32).reshape(1, -1)
        ssl_scaled = self.ssl_scaler.transform(ssl_features).flatten()
        
        # Extract and scale Bio features for this row
        bio_features = row[self.bio_cols].values.astype(np.float32).reshape(1, -1)
        bio_scaled = self.bio_scaler.transform(bio_features).flatten()
        
        label = np.float32(row["label"])
        
        return torch.tensor(ssl_scaled), torch.tensor(bio_scaled), torch.tensor(label)

def main():
    print("Loading Scalers...")
    ssl_scaler = joblib.load(SSL_SCALER_PATH)
    bio_scaler = joblib.load(BIO_SCALER_PATH)

    print("Loading Models...")
    # Initialize networks based on the number of features they were trained on
    ssl_model = SSLModel(input_dim=1536)
    ssl_model.load_state_dict(torch.load(SSL_WEIGHTS_PATH))
    ssl_model.eval()

    bio_model = BioModel(input_dim=14)
    bio_model.load_state_dict(torch.load(BIO_WEIGHTS_PATH))
    bio_model.eval()
    
    print("Loading Evaluation datasets...")
    ssl_df = pd.read_csv(SSL_CSV)
    bio_df = pd.read_csv(BIO_CSV).drop(columns=["label", "codec"])
    
    print("Merging datasets to ensure perfect alignment...")
    eval_df = pd.merge(ssl_df, bio_df, on="filename", how="inner")
    
    # Free up RAM immediately
    del ssl_df
    del bio_df
    
    print("Applying Biological feature engineering...")
    eval_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    eval_df.dropna(inplace=True)
    eval_df["f0_cov"] = eval_df["stdevF0"] / (eval_df["meanF0"] + 1e-6)
    
    jsh_cols = ["j_local", "j_abs", "j_rap", "j_ppq5", "j_ddp", "s_local", "s_db", "s_apq3", "s_apq5", "s_apq11", "s_dda"]
    for col in jsh_cols:
        eval_df[col] = np.log(np.clip(eval_df[col].astype(float), 1e-6, None))

    # Identify exact columns for each model
    ssl_cols = [c for c in eval_df.columns if c.startswith("ssl_")]
    bio_cols = ["meanF0", "f0_cov", "hnr"] + jsh_cols
    
    datasets = {
        "Clean (No Codec) Audio": eval_df[eval_df['codec'] == 'nocodec'],
        "Compressed Audio": eval_df[eval_df['codec'] != 'nocodec']
    }
    
    for condition, df in datasets.items():
        if len(df) == 0: continue
            
        print(f"\n========================================")
        print(f"Evaluating {condition} ({len(df)} files)")
        print(f"========================================")
        
        dataset = MultiModelEvalDataset(df, ssl_scaler, bio_scaler, ssl_cols, bio_cols)
        # num_workers=0 bypasses the Windows multiprocessing crash
        loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=0)
        
        all_ssl_scores, all_bio_scores, all_labels = [], [], []
        
        print("Running simultaneous inference...")
        with torch.no_grad():
            for batch_ssl, batch_bio, batch_labels in tqdm(loader, desc="Inference"):
                # Run both models
                ssl_logits = ssl_model(batch_ssl)
                bio_logits = bio_model(batch_bio)
                
                # Convert logits to 0-1 percentage scores using Sigmoid
                ssl_scores = torch.sigmoid(ssl_logits).numpy().flatten()
                bio_scores = torch.sigmoid(bio_logits).numpy().flatten()
                
                all_ssl_scores.extend(ssl_scores)
                all_bio_scores.extend(bio_scores)
                all_labels.extend(batch_labels.numpy())
                
        all_ssl_scores = np.array(all_ssl_scores)
        all_bio_scores = np.array(all_bio_scores)
        all_labels = np.array(all_labels)

        # Base Model Check
        ssl_eer = calculate_eer(all_labels, all_ssl_scores)
        bio_eer = calculate_eer(all_labels, all_bio_scores)
        print(f"\nStandalone SSL EER: {ssl_eer:.2f}%")
        print(f"Standalone Bio EER: {bio_eer:.2f}%")
        print("\n--- Running Grid Search for Optimal Weights ---")
        
        best_eer = 100.0
        best_ssl_weight = 0.0
        
        # Test all weight combinations in 5% increments
        for w in np.arange(0.0, 1.05, 0.05):
            ssl_weight = w
            bio_weight = 1.0 - w
            
            # The Weighted Sum Rule
            fused_scores = (ssl_weight * all_ssl_scores) + (bio_weight * all_bio_scores)
            eer = calculate_eer(all_labels, fused_scores)
            
            if eer < best_eer:
                best_eer = eer
                best_ssl_weight = ssl_weight
                
        print(f"\n🏆 BEST SCORE-LEVEL FUSION EER: {best_eer:.2f}%")
        print(f"🏆 OPTIMAL WEIGHTS: {best_ssl_weight:.2f} SSL / {1.0 - best_ssl_weight:.2f} Bio")

if __name__ == "__main__":
    main()