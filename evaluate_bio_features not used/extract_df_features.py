import os
import pandas as pd
import numpy as np
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- Paths based on your folder structure ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the DF keys
DF_KEYS_FILE = os.path.join(
    BASE_DIR, 
    "data", "DF-keys-full", "keys", "DF", "CM", "trial_metadata.txt"
)

# Path to the directory containing the flac files (assuming part00-part03 are extracted to one folder)
FLAC_DIR = os.path.join(
    BASE_DIR, 
    "data", "DF", "ASVspoof2021_DF_eval", "flac"
)

OUTPUT_CSV = os.path.join(BASE_DIR, "eval_df_features_9_attrs.csv")

# --- Feature Extraction Function ---
def extract_jsh_9_attrs(audio_path, f0min=75, f0max=600):
    try:
        sound = parselmouth.Sound(audio_path)
        point_process = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
        
        j_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        j_abs   = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        j_rap   = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        j_ppq5  = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        
        s_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        s_db    = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        s_apq3  = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        s_apq5  = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        hnr = call(harmonicity, "Get mean", 0, 0)
        return j_local, j_abs, j_rap, j_ppq5, s_local, s_db, s_apq3, s_apq5, hnr
    except Exception:
        return (np.nan,) * 9

# --- Multiprocessing Worker Function ---
def process_file(row_dict):
    # 'row_dict' is now a standard Python dictionary instead of a pandas namedtuple
    file_id = row_dict["filename"]
    target = row_dict["target"]
    codec = row_dict["codec"]  
    
    audio_path = os.path.join(FLAC_DIR, f"{file_id}.flac")
    if os.path.exists(audio_path):
        features = extract_jsh_9_attrs(audio_path)
        return {
            "filename": file_id, "codec": codec, "label": target,
            "j_local": features[0], "j_abs": features[1], "j_rap": features[2],
            "j_ppq5": features[3], "s_local": features[4], "s_db": features[5],
            "s_apq3": features[6], "s_apq5": features[7], "hnr": features[8]
        }
    return None

def main():
    print("Loading ASVspoof 2021 DF protocol file...")
    
    # Use regex whitespace separator '\s+' to safely handle any spacing
    labels_df = pd.read_csv(DF_KEYS_FILE, sep=r'\s+', header=None)
    
    # Explicitly map the exact column indices for ASVspoof 2021 DF
    # Format: [0:Speaker] [1:Filename] [2:Codec] [3:Task] [4:Attack] [5:Label] [6:Trim] [7:Phase]
    labels_df["filename"] = labels_df[1]
    labels_df["codec"] = labels_df[2]
    labels_df["target"] = labels_df[5].apply(lambda x: 1 if x == "bonafide" else 0)
    
    # Drop everything else
    labels_df = labels_df[["filename", "target", "codec"]]
    
    tasks = labels_df.to_dict('records')
    
    # --- QUICK DEBUG CHECK ---
    # Print the first file path to prove it works before running all 600k
    test_path = os.path.join(FLAC_DIR, f'{tasks[0]["filename"]}.flac')
    print(f"\nDEBUG: Looking for first file at:\n{test_path}")
    print(f"DEBUG: Does first file exist? {os.path.exists(test_path)}\n")
    
    if not os.path.exists(test_path):
        print("ERROR: File not found! Check your folder structure. Exiting...")
        return
    # -------------------------
    
    print(f"Starting feature extraction for {len(tasks)} files using {cpu_count()} CPU cores...")
    
    # Run multiprocessing on the list of dictionaries
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, tasks), total=len(tasks)))
    
    # Filter out any files that were not found
    features_list = [r for r in results if r is not None]
    print(f"\nSuccessfully extracted features for {len(features_list)} files.")
    
    if len(features_list) > 0:
        df_features = pd.DataFrame(features_list)
        df_features.fillna(0, inplace=True) 
        df_features.to_csv(OUTPUT_CSV, index=False)
        print(f"Extraction complete! Features saved to {OUTPUT_CSV}")
    else:
        print("Failed to extract any files. CSV not saved.")

if __name__ == "__main__":
    main()