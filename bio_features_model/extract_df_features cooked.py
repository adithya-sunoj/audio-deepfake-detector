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
# Establishes the relative base directory to ensure the code runs on any machine
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The protocol file contains the ground truth (real vs fake) and degradation data (codec)
DF_KEYS_FILE = os.path.join(BASE_DIR, "data", "DF-keys-full", "keys", "DF", "CM", "trial_metadata.txt")
FLAC_DIR = os.path.join(BASE_DIR, "data", "DF", "ASVspoof2021_DF_eval", "flac")
OUTPUT_CSV = os.path.join(BASE_DIR, "eval_df_bio_academic.csv")

# This is the crucial memory-management threshold. RAM is dumped to disk every 50k files.
SAVE_INTERVAL = 50000


# --- Librosa VAD (Voice Activity Detection) ---
def apply_librosa_vad(y, sr=16000):
    """
    Strips dead silence from audio files. Biological features (jitter/shimmer) only 
    exist during active vocal cord vibrations. Measuring silence corrupts the metrics.
    """
    # top_db=30 is a conservative threshold; anything 30 decibels quieter than the peak 
    # is classified as silence. frame_length and hop_length control the resolution.
    intervals = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=512)
    
    if len(intervals) == 0: return np.array([])
    
    # Concatenates all the actively spoken segments back into one continuous array
    voiced_frames = [y[start:end] for start, end in intervals]
    return np.concatenate(voiced_frames)


# --- Praat Extraction ---
def measure_pitch_jsh(audio_path, f0min=75, f0max=600):
    """
    Extracts 14 biological acoustic features using the Praat engine. 
    f0min (75Hz) and f0max (600Hz) restrict the algorithm to the bounds of normal human speech, 
    preventing it from mistakenly analyzing high-frequency background noise.
    """
    try:
        y, sr = sf.read(audio_path, dtype='float32')
        
        # Standardization: Ensure all audio is evaluated at exactly 16,000 Hz
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Strip silence
        y_voiced = apply_librosa_vad(y, sr)
        
        # If the file contains less than 100 milliseconds of speech after VAD, 
        # it is too short to mathematically calculate pitch variance. Return NaNs.
        if len(y_voiced) < sr * 0.1: return (np.nan,) * 14
            
        # PEAK NORMALIZATION: Shimmer measures amplitude (volume) instability. 
        # If a file was recorded too quietly, Praat will struggle to find the peaks.
        # This scales the loudest peak up to exactly 1.0 without distorting the wave.
        max_val = np.max(np.abs(y_voiced))
        if max_val > 0: y_voiced = y_voiced / max_val
            
        # Convert numpy array into the specific Sound object required by the Praat C++ engine
        sound = parselmouth.Sound(y_voiced, sr)
        
        # Generate the underlying mathematical models of the voice
        pitch = call(sound, "To Pitch (cc)", 0.0, f0min, 15, "yes", 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
        point_process = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
        
        # Extract general vocal tract measurements
        meanF0 = call(pitch, "Get mean", 0, 0, "Hertz")
        stdevF0 = call(pitch, "Get standard deviation", 0 ,0, "Hertz")
        hnr = call(harmonicity, "Get mean", 0, 0)
        
        # Extract 5 variants of Jitter (Frequency instability) and 6 variants 
        # of Shimmer (Amplitude instability) to capture micro-imperfections in the voice.
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
        
        # Praat returns the string "undefined" if an algorithm mathematically fails on a file.
        # This list comprehension safely converts those strings into standard numpy NaNs.
        return tuple(np.nan if str(f) == "undefined" else float(f) for f in features)
    except Exception:
        # Fallback to prevent a single corrupted audio file from crashing the whole pipeline
        return (np.nan,) * 14


# --- Multiprocessing Worker Function ---
def process_file(row_dict):
    """
    Self-contained worker designed to be fired concurrently across multiple CPU cores.
    Packages metadata (like codec) alongside the 14 extracted features.
    """
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
    # sep=r'\s+' safely handles protocol files where spacing varies between columns
    labels_df = pd.read_csv(DF_KEYS_FILE, sep=r'\s+', header=None)
    labels_df["filename"] = labels_df[1]
    labels_df["codec"] = labels_df[2]
    # Convert string targets into binary labels for the neural network
    labels_df["target"] = labels_df[5].apply(lambda x: 1 if x == "bonafide" else 0)
    
    # --- Checkpointing & Resume Logic ---
    # Processing 600k+ files takes days. This scans the disk for existing checkpoints
    # so execution can resume exactly where it left off in case of an interruption.
    processed_filenames = set()
    ckpt_files = sorted(glob.glob(OUTPUT_CSV.replace(".csv", "_ckpt*.csv")))
    for f in ckpt_files:
        df_ckpt = pd.read_csv(f, usecols=["filename"])
        processed_filenames.update(df_ckpt["filename"].tolist())
        
    if processed_filenames:
        print(f"[Resume] Found {len(processed_filenames)} already processed files.")
        # Drop already processed files from the dataframe
        labels_df = labels_df[~labels_df["filename"].isin(processed_filenames)]
        
    if len(labels_df) == 0:
        print("All files processed!")
        merge_checkpoints(ckpt_files)
        return

    # Convert the remaining rows into independent dictionaries for multiprocessing
    tasks = labels_df[["filename", "codec", "target"]].to_dict('records')
    
    # Detect available hardware to maximize extraction speed
    cores = cpu_count()
    checkpoint_num = len(ckpt_files) + 1
    feature_rows = []
    
    print(f"Extracting DF VAD features using {cores} CPU cores...")
    
    # Initialize the worker pool
    with Pool(processes=cores) as pool:
        # pool.imap_unordered ensures that as soon as any CPU core finishes a file, 
        # it is yielded immediately without waiting for slower files to finish first.
        for result in tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks)):
            if result is not None:
                feature_rows.append(result)
                
            # --- Dynamic RAM Clearing ---
            # Holding 600k rows in a Python list will cause an Out-Of-Memory (OOM) crash.
            # Once the buffer hits 50k, it flushes to a physical disk file.
            if len(feature_rows) >= SAVE_INTERVAL:
                temp_csv = OUTPUT_CSV.replace(".csv", f"_ckpt{checkpoint_num}.csv")
                pd.DataFrame(feature_rows).to_csv(temp_csv, index=False)
                
                # Instantly frees up the system RAM for the next batch
                feature_rows = []
                checkpoint_num += 1

    # Catch and save any final stragglers that didn't hit the 50k threshold
    if feature_rows:
        temp_csv = OUTPUT_CSV.replace(".csv", f"_ckpt{checkpoint_num}.csv")
        pd.DataFrame(feature_rows).to_csv(temp_csv, index=False)

    # Begin assembly of the final dataset
    all_ckpt_files = sorted(glob.glob(OUTPUT_CSV.replace(".csv", "_ckpt*.csv")))
    merge_checkpoints(all_ckpt_files)


def merge_checkpoints(ckpt_files):
    """
    RAM-Safe merging. Loads and appends files chunk by chunk instead of 
    attempting to hold the entire merged dataset in memory at once.
    """
    print("\nMerging checkpoints...")
    first = True
    for f in tqdm(ckpt_files):
        chunk = pd.read_csv(f)
        
        # On-the-fly data cleaning: Drop files that lacked enough voice data
        chunk.dropna(subset=['j_local'], inplace=True) 
        
        # Safely fill isolated NaN values with the mean to prevent model crashes later
        chunk.fillna(chunk.mean(numeric_only=True), inplace=True) 
        
        # 'w' overwrites for the very first file; 'a' appends to the bottom for the rest
        chunk.to_csv(OUTPUT_CSV, mode='w' if first else 'a', header=first, index=False)
        first = False
        
    print(f"Final CSV saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()