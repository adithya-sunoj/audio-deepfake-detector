import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import parselmouth
import glob
from parselmouth.praat import call
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DF_KEYS_FILE = os.path.join(BASE_DIR, "data", "DF-keys-full", "keys", "DF", "CM", "trial_metadata.txt")
FLAC_DIR = os.path.join(BASE_DIR, "data", "DF", "ASVspoof2021_DF_eval", "flac")
OUTPUT_CSV = os.path.join(BASE_DIR, "eval_df_bio_academic.csv")

SAVE_INTERVAL = 50000

# --- Librosa VAD ---
def apply_librosa_vad(y, sr=16000):
    intervals = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=512)
    if len(intervals) == 0: return np.array([])
    voiced_frames = [y[start:end] for start, end in intervals]
    return np.concatenate(voiced_frames)

# --- Praat Extraction ---
def measure_pitch_jsh(audio_path, f0min=75, f0max=600):
    try:
        y, sr = sf.read(audio_path, dtype='float32')
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        y_voiced = apply_librosa_vad(y, sr)
        if len(y_voiced) < sr * 0.1: return (np.nan,) * 14
            
        max_val = np.max(np.abs(y_voiced))
        if max_val > 0: y_voiced = y_voiced / max_val
            
        sound = parselmouth.Sound(y_voiced, sr)
        
        pitch = call(sound, "To Pitch (cc)", 0.0, f0min, 15, "yes", 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
        point_process = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
        
        meanF0 = call(pitch, "Get mean", 0, 0, "Hertz")
        stdevF0 = call(pitch, "Get standard deviation", 0 ,0, "Hertz")
        hnr = call(harmonicity, "Get mean", 0, 0)
        
        features = [
            meanF0, stdevF0, hnr, 
            call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
            call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3),
            call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3),
            call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3),
            call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3),
            call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        ]
        return tuple(np.nan if str(f) == "undefined" else float(f) for f in features)
    except Exception:
        return (np.nan,) * 14

def process_file(row_dict):
    file_id = row_dict["filename"]
    codec = row_dict["codec"]
    target = row_dict["target"]
    audio_path = os.path.join(FLAC_DIR, f"{file_id}.flac")
    
    if os.path.exists(audio_path):
        feats = measure_pitch_jsh(audio_path)
        return {
            "filename": file_id, "codec": codec, "label": target,
            "meanF0": feats[0], "stdevF0": feats[1], "hnr": feats[2],
            "j_local": feats[3], "j_abs": feats[4], "j_rap": feats[5], "j_ppq5": feats[6], "j_ddp": feats[7],
            "s_local": feats[8], "s_db": feats[9], "s_apq3": feats[10], "s_apq5": feats[11], "s_apq11": feats[12], "s_dda": feats[13]
        }
    return None

def main():
    print("Loading DF Protocol file...")
    labels_df = pd.read_csv(DF_KEYS_FILE, sep=r'\s+', header=None)
    labels_df["filename"] = labels_df[1]
    labels_df["codec"] = labels_df[2]
    labels_df["target"] = labels_df[5].apply(lambda x: 1 if x == "bonafide" else 0)
    
    # Resume Logic
    processed_filenames = set()
    ckpt_files = sorted(glob.glob(OUTPUT_CSV.replace(".csv", "_ckpt*.csv")))
    for f in ckpt_files:
        df_ckpt = pd.read_csv(f, usecols=["filename"])
        processed_filenames.update(df_ckpt["filename"].tolist())
        
    if processed_filenames:
        print(f"[Resume] Found {len(processed_filenames)} already processed files.")
        labels_df = labels_df[~labels_df["filename"].isin(processed_filenames)]
        
    if len(labels_df) == 0:
        print("All files processed!")
        merge_checkpoints(ckpt_files)
        return

    tasks = labels_df[["filename", "codec", "target"]].to_dict('records')
    cores = cpu_count()
    checkpoint_num = len(ckpt_files) + 1
    feature_rows = []
    
    print(f"Extracting DF VAD features using {cores} CPU cores...")
    
    with Pool(processes=cores) as pool:
        for result in tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks)):
            if result is not None:
                feature_rows.append(result)
                
            if len(feature_rows) >= SAVE_INTERVAL:
                temp_csv = OUTPUT_CSV.replace(".csv", f"_ckpt{checkpoint_num}.csv")
                pd.DataFrame(feature_rows).to_csv(temp_csv, index=False)
                feature_rows = []
                checkpoint_num += 1

    if feature_rows:
        temp_csv = OUTPUT_CSV.replace(".csv", f"_ckpt{checkpoint_num}.csv")
        pd.DataFrame(feature_rows).to_csv(temp_csv, index=False)

    all_ckpt_files = sorted(glob.glob(OUTPUT_CSV.replace(".csv", "_ckpt*.csv")))
    merge_checkpoints(all_ckpt_files)

def merge_checkpoints(ckpt_files):
    print("\nMerging checkpoints...")
    first = True
    for f in tqdm(ckpt_files):
        chunk = pd.read_csv(f)
        chunk.dropna(subset=['j_local'], inplace=True) # Drop silence
        chunk.fillna(chunk.mean(numeric_only=True), inplace=True) # Fill stragglers safely
        chunk.to_csv(OUTPUT_CSV, mode='w' if first else 'a', header=first, index=False)
        first = False
    print(f"Final CSV saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()