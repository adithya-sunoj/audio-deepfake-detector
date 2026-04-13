import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- 1. Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTOCOL_FILE = os.path.join(BASE_DIR, "data", "LA", "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")
FLAC_DIR = os.path.join(BASE_DIR, "data", "LA", "ASVspoof2019_LA_train", "flac")
OUTPUT_CSV = os.path.join(BASE_DIR, "train_bio_features_academic.csv")

# --- 2. Pure Python VAD (Using Librosa Split) ---
def apply_librosa_vad(y, sr=16000):
    """
    Finds all non-silent intervals in the audio and concatenates them.
    Solves the 'dead silence in the middle of a sentence' problem without needing C++ compilers.
    """
    # top_db=30 is a very safe threshold for speech vs absolute silence/static
    intervals = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=512)
    
    if len(intervals) == 0:
        return np.array([])
        
    voiced_frames = []
    for interval in intervals:
        start, end = interval
        voiced_frames.append(y[start:end])
        
    return np.concatenate(voiced_frames)

# --- 3. Academic Praat Extraction ---
def measure_pitch_jsh(audio_path, f0min=75, f0max=600):
    try:
        y, sr = sf.read(audio_path, dtype='float32')
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        # VAD Filter
        y_voiced = apply_librosa_vad(y, sr)
        if len(y_voiced) < sr * 0.1: # Less than 100ms of speech
            return (np.nan,) * 14
            
        # PEAK NORMALIZATION: Ensure Praat reads loud, clear peaks for Shimmer
        max_val = np.max(np.abs(y_voiced))
        if max_val > 0:
            y_voiced = y_voiced / max_val
            
        sound = parselmouth.Sound(y_voiced, sr)
        
        # Optimized Pitch Tracking (Cross-Correlation)
        pitch = call(sound, "To Pitch (cc)", 0.0, f0min, 15, "yes", 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
        point_process = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
        
        meanF0 = call(pitch, "Get mean", 0, 0, "Hertz")
        stdevF0 = call(pitch, "Get standard deviation", 0 ,0, "Hertz")
        hnr = call(harmonicity, "Get mean", 0, 0)
        
        localJitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        localabsoluteJitter = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rapJitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq5Jitter = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        ddpJitter = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
        
        localShimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        localdbShimmer = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq3Shimmer = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        aqpq5Shimmer = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq11Shimmer = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        ddaShimmer = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        features = [meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter,
                    localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer]
        
        return tuple(np.nan if str(f) == "undefined" else float(f) for f in features)
        
    except Exception:
        return (np.nan,) * 14

# --- 4. Multiprocessing Worker Function ---
def process_file(row_dict):
    file_id = row_dict["filename"]
    target = row_dict["target"]
    audio_path = os.path.join(FLAC_DIR, f"{file_id}.flac")
    
    if os.path.exists(audio_path):
        feats = measure_pitch_jsh(audio_path)
        return {
            "filename": file_id, "label": target,
            "meanF0": feats[0], "stdevF0": feats[1], "hnr": feats[2],
            "j_local": feats[3], "j_abs": feats[4], "j_rap": feats[5], "j_ppq5": feats[6], "j_ddp": feats[7],
            "s_local": feats[8], "s_db": feats[9], "s_apq3": feats[10], "s_apq5": feats[11], "s_apq11": feats[12], "s_dda": feats[13]
        }
    return None

# --- 5. Main Execution ---
def main():
    print("Loading ASVspoof 2019 LA protocol file...")
    labels_df = pd.read_csv(PROTOCOL_FILE, sep=" ", header=None, names=["speaker_id", "filename", "env", "attack", "label"])
    labels_df["target"] = labels_df["label"].apply(lambda x: 1 if x == "bonafide" else 0)
    
    tasks = labels_df[["filename", "target"]].to_dict('records')
    cores = cpu_count()
    
    print(f"Extracting VAD features (Librosa engine) using {cores} CPU cores...")
    with Pool(processes=cores) as pool:
        results = list(tqdm(pool.imap(process_file, tasks), total=len(tasks)))
        
    features_list = [r for r in results if r is not None]
    df = pd.DataFrame(features_list)
    initial_len = len(df)
    
    df.dropna(subset=['j_local'], inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    print(f"\nDropped {initial_len - len(df)} files that were entirely silent or unvoiced.")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Academic features saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()