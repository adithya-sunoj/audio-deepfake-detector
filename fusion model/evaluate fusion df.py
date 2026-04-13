import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve
from tqdm import tqdm

# --- 1. Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALER_PATH = os.path.join(BASE_DIR, "fusion_academic_scaler.pkl")
WEIGHTS_PATH = os.path.join(BASE_DIR, "fusion_academic_mlp_weights.pt")
SSL_CSV = os.path.join(BASE_DIR, "eval_df_ssl_features.csv")
BIO_CSV = os.path.join(BASE_DIR, "eval_df_bio_academic.csv")

# --- 2. Architecture ---
class FusionMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

def calculate_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return ((fpr[idx] + fnr[idx]) / 2.0) * 100

# --- 3. RAM-Safe PyTorch Dataset ---
class DFEvalDataset(Dataset):
    def __init__(self, dataframe, scaler, feature_cols):
        # We store the dataframe reference, but we DO NOT call .values on the whole thing
        self.df = dataframe
        self.scaler = scaler
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Only extract the specific row needed for this exact moment
        row = self.df.iloc[idx]
        features = row[self.feature_cols].values.astype(np.float32).reshape(1, -1)
        
        # Scale just this one row
        scaled_features = self.scaler.transform(features).flatten()
        label = np.float32(row["label"])
        
        return torch.tensor(scaled_features), torch.tensor(label)

def main():
    print("Loading Fusion Scaler...")
    try:
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print(f"ERROR: Could not find scaler at {SCALER_PATH}")
        return

    model = FusionMLP(1550)
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    model.eval()
    
    print("Loading SSL and Bio Evaluation datasets...")
    # Load chunks to save memory during merge
    ssl_df = pd.read_csv(SSL_CSV)
    bio_df = pd.read_csv(BIO_CSV).drop(columns=["label", "codec"])
    
    print("Fusing evaluation datasets...")
    eval_df = pd.merge(ssl_df, bio_df, on="filename", how="inner")
    
    # Delete massive raw dataframes from RAM to free up space immediately
    del ssl_df
    del bio_df
    
    print("Applying feature engineering...")
    eval_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    eval_df.dropna(inplace=True)
    eval_df["f0_cov"] = eval_df["stdevF0"] / (eval_df["meanF0"] + 1e-6)
    
    jsh_cols = ["j_local", "j_abs", "j_rap", "j_ppq5", "j_ddp", "s_local", "s_db", "s_apq3", "s_apq5", "s_apq11", "s_dda"]
    for col in jsh_cols:
        eval_df[col] = np.log(np.clip(eval_df[col].astype(float), 1e-6, None))

    ssl_cols = [c for c in eval_df.columns if c.startswith("ssl_")]
    feature_cols = ssl_cols + ["meanF0", "f0_cov", "hnr"] + jsh_cols
    
    datasets = {
        "Clean (No Codec)": eval_df[eval_df['codec'] == 'nocodec'],
        "Compressed": eval_df[eval_df['codec'] != 'nocodec']
    }
    
    for condition, df in datasets.items():
        if len(df) == 0: continue
            
        print(f"\nEvaluating {condition} Audio ({len(df)} files)...")
        
        # Stream data row-by-row using DataLoader
        dataset = DFEvalDataset(df, scaler, feature_cols)
        loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=0)
         
        all_scores, all_labels = [], []
        
        with torch.no_grad():
            for batch_features, batch_labels in tqdm(loader, desc="Inference"):
                logits = model(batch_features)
                scores = torch.sigmoid(logits).numpy().flatten()
                
                all_scores.extend(scores)
                all_labels.extend(batch_labels.numpy())
                
        eer = calculate_eer(all_labels, all_scores)
        print(f"Equal Error Rate (EER): {eer:.2f}%")

if __name__ == "__main__":
    main()