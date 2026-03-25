import os
import pandas as pd
import numpy as np
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm

# --- 1. Define Paths based on your folder structure ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the label/protocol file
PROTOCOL_FILE = os.path.join(
    BASE_DIR, 
    "data", "LA", "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt"
)

# Path to the directory containing the flac files
FLAC_DIR = os.path.join(
    BASE_DIR, 
    "data", "LA", "ASVspoof2019_LA_train", "flac"
)

# Output CSV to save your features
OUTPUT_CSV = os.path.join(BASE_DIR, "train_bio_features_9_attrs.csv")

# --- 2. Define the Feature Extraction Function ---
def extract_jsh_9_attrs(audio_path, f0min=75, f0max=600):
    """
    Extracts 4 Jitter, 4 Shimmer, and 1 HNR attributes using Praat's Parselmouth.
    """
    try:
        # Load the sound file
        sound = parselmouth.Sound(audio_path)
        
        # Create objects needed for extraction
        point_process = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
        
        # --- 4 Jitter Variations ---
        j_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        j_abs   = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        j_rap   = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        j_ppq5  = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        
        # --- 4 Shimmer Variations ---
        s_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        s_db    = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        s_apq3  = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        s_apq5  = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        # --- 1 Harmonics-to-Noise Ratio ---
        hnr = call(harmonicity, "Get mean", 0, 0)
        
        return j_local, j_abs, j_rap, j_ppq5, s_local, s_db, s_apq3, s_apq5, hnr
        
    except Exception as e:
        # If the audio contains no pitch/voice, Praat fails. Return 9 NaNs.
        return (np.nan,) * 9

# --- 3. Load Labels and Process Files ---
def main():
    print("Loading ASVspoof 2019 LA protocol file...")
    
    # The protocol file is space-separated with 5 columns
    labels_df = pd.read_csv(
        PROTOCOL_FILE, sep=" ", header=None, 
        names=["speaker_id", "filename", "env", "attack", "label"]
    )
    
    # Filter just what we need
    labels_df = labels_df[["filename", "label"]]
    
    # Convert 'bonafide' (real) to 1, and 'spoof' (fake) to 0
    labels_df["target"] = labels_df["label"].apply(lambda x: 1 if x == "bonafide" else 0)
    
    features_list = []
    
    print(f"Starting feature extraction for {len(labels_df)} files...")
    
    # Iterate through each row in the protocol dataframe
    for index, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        file_id = row["filename"]
        target = row["target"]
        
        # Construct the exact file path
        audio_path = os.path.join(FLAC_DIR, f"{file_id}.flac")
        
        if os.path.exists(audio_path):
            # Extract the 9 features
            j_local, j_abs, j_rap, j_ppq5, s_local, s_db, s_apq3, s_apq5, hnr = extract_jsh_9_attrs(audio_path)
            
            features_list.append({
                "filename": file_id,
                "j_local": j_local,
                "j_abs": j_abs,
                "j_rap": j_rap,
                "j_ppq5": j_ppq5,
                "s_local": s_local,
                "s_db": s_db,
                "s_apq3": s_apq3,
                "s_apq5": s_apq5,
                "hnr": hnr,
                "label": target
            })
        else:
            print(f"Warning: File not found {audio_path}")

    # --- 4. Save to Tabular Format ---
    df_features = pd.DataFrame(features_list)
    
    # Handle files that yielded NaN due to being unvoiced
    df_features.fillna(0, inplace=True) 
    
    df_features.to_csv(OUTPUT_CSV, index=False)
    print(f"\nExtraction complete! Features saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()