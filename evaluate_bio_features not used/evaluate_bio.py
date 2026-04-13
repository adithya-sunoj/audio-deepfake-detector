import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d

# --- 1. Re-declare the Network Architecture ---
class BioMLP(nn.Module):
    def __init__(self, input_dim):
        super(BioMLP, self).__init__()
        # A simple, lightweight feed-forward network
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16), # Normalizes data to speed up training
            nn.Dropout(0.3),    # Prevents overfitting
            
            nn.Linear(16, 8),
            nn.ReLU(),
            
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)

# --- 2. Function to Calculate EER ---
def calculate_eer(y_true, y_scores):
    """Calculates Equal Error Rate using ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    frr = 1 - tpr
    
    # Find the point where False Positive Rate (FPR) == False Rejection Rate (FRR)
    eer_threshold = thresholds[np.nanargmin(np.absolute(fpr - frr))]
    eer = fpr[np.nanargmin(np.absolute(fpr - frr))]
    return eer * 100 # return as percentage

def main():
    # 1. Load the Model
    input_dim = 9
    model = BioMLP(input_dim)
    model.load_state_dict(torch.load("bio_mlp_weights.pt"))
    model.eval() # Set to evaluation mode
    
    # 2. We must re-fit the scaler on the TRAINING data so the scales match
    # Do not fit it on the evaluation data!
    print("Loading training data to fit scaler...")
    train_df = pd.read_csv("train_bio_features_9_attrs.csv")
    X_train = train_df[['j_local', 'j_abs', 'j_rap', 'j_ppq5', 's_local', 's_db', 's_apq3', 's_apq5', 'hnr']].values
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # 3. Load the Evaluation Data
    print("Loading evaluation data...")
    eval_df = pd.read_csv("eval_df_features_9_attrs.csv")
    
    # Separate into Clean (nocodec) and Compressed (everything else)
    clean_df = eval_df[eval_df['codec'] == 'nocodec']
    compressed_df = eval_df[eval_df['codec'] != 'nocodec']
    
    datasets = {"Clean": clean_df, "Compressed": compressed_df}
    
    for condition, df in datasets.items():
        X_eval = df[['j_local', 'j_abs', 'j_rap', 'j_ppq5', 's_local', 's_db', 's_apq3', 's_apq5', 'hnr']].values
        y_eval = df['label'].values
        
        # Scale the features using the scaler fitted on training data
        X_eval_scaled = scaler.transform(X_eval)
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_eval_scaled, dtype=torch.float32)
        
        # 4. Get Predictions
        with torch.no_grad():
            outputs = model(X_tensor)
            # Since the model outputs logits, apply sigmoid to get probabilities
            probabilities = torch.sigmoid(outputs).numpy().flatten()
            
        # Calculate and print EER
        eer = calculate_eer(y_eval, probabilities)
        print(f"\n--- Condition: {condition} Audio ---")
        print(f"Files tested: {len(df)}")
        print(f"Equal Error Rate (EER): {eer:.2f}%")

if __name__ == "__main__":
    main()