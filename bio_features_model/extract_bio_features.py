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
# Dynamically determine the root directory of the project to ensure relative paths work on any machine.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the location of the dataset's protocol file, which acts as a master spreadsheet linking filenames to their true labels (real vs fake).
PROTOCOL_FILE = os.path.join(BASE_DIR, "data", "LA", "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")

# Define the directory where the raw .flac audio files are stored.
FLAC_DIR = os.path.join(BASE_DIR, "data", "LA", "ASVspoof2019_LA_train", "flac")

# Define where the final extracted biological features should be saved.
OUTPUT_CSV = os.path.join(BASE_DIR, "train_bio_features_academic.csv")


# --- 2. Pure Python VAD (Using Librosa Split) ---
def apply_librosa_vad(y, sr=16000):
    """
    Finds all non-silent intervals in the audio and concatenates them.
    Solves the 'dead silence in the middle of a sentence' problem without needing C++ compilers.
    """
    # Use librosa to split the audio. top_db=30 means anything 30 decibels quieter than the peak volume is considered silence.
    # frame_length and hop_length define the resolution of the sliding window used to detect the silence.
    intervals = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=512)
    
    # If the entire file is silent, return an empty array to prevent downstream math errors.
    if len(intervals) == 0:
        return np.array([])
        
    voiced_frames = []
    # Loop through the detected speech intervals and collect the raw audio data for those specific timeframes.
    for interval in intervals:
        start, end = interval
        voiced_frames.append(y[start:end])
        
    # Stitch the isolated speech chunks back together into one continuous array of uninterrupted talking.
    return np.concatenate(voiced_frames)


# --- 3. Academic Praat Extraction ---
def measure_pitch_jsh(audio_path, f0min=75, f0max=600):
    # Wrap in a try-except block because external linguistics libraries can occasionally fail on corrupted audio files.
    try:
        # Load the raw audio data as float32, which is the standard datatype for digital signal processing.
        y, sr = sf.read(audio_path, dtype='float32')
        
        # Ensure strict uniformity. If the audio is not 16kHz, resample it mathematically so all files are evaluated equally.
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        # VAD Filter: Apply the Voice Activity Detection function to remove silence.
        y_voiced = apply_librosa_vad(y, sr)
        
        # If the resulting speech is less than 100 milliseconds, there isn't enough data for a statistically significant Praat reading.
        if len(y_voiced) < sr * 0.1: # Less than 100ms of speech
            return (np.nan,) * 14
            
        # PEAK NORMALIZATION: Ensure Praat reads loud, clear peaks for Shimmer calculation.
        # Find the absolute loudest point in the array.
        max_val = np.max(np.abs(y_voiced))
        # If the audio isn't entirely dead silence, divide the whole array by the max value, artificially boosting the loudest peak to exactly 1.0.
        if max_val > 0:
            y_voiced = y_voiced / max_val
            
        # Convert the normalized numpy array into a specialized parselmouth Sound object required for Praat analysis.
        sound = parselmouth.Sound(y_voiced, sr)
        
        # Optimized Pitch Tracking (Cross-Correlation): Execute Praat's C++ algorithm to track the fundamental frequency (pitch).
        # We specify f0min=75 and f0max=600 to restrict the search exclusively to the normal range of human vocal cords.
        pitch = call(sound, "To Pitch (cc)", 0.0, f0min, 15, "yes", 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
        
        # Create a point process to precisely map the peaks of the sound waves, which is required to measure jitter and shimmer.
        point_process = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        
        # Calculate the ratio of harmonic sound (human voice) to non-harmonic sound (static/noise).
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
        
        # Extract the basic statistical properties of the pitch and harmonicity.
        meanF0 = call(pitch, "Get mean", 0, 0, "Hertz")
        stdevF0 = call(pitch, "Get standard deviation", 0 ,0, "Hertz")
        hnr = call(harmonicity, "Get mean", 0, 0)
        
        # Extract five different mathematical variations of Jitter (the micro-instability of the voice's frequency).
        localJitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        localabsoluteJitter = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rapJitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq5Jitter = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        ddpJitter = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
        
        # Extract six different mathematical variations of Shimmer (the micro-instability of the voice's volume/amplitude).
        localShimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        localdbShimmer = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq3Shimmer = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        aqpq5Shimmer = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq11Shimmer = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        ddaShimmer = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        # Bundle all 14 biological features into a single list for organized return.
        features = [meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter,
                    localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer]
        
        # Safely return the features. If Praat returns a string "undefined" due to bad audio math, convert it to a float NaN.
        return tuple(np.nan if str(f) == "undefined" else float(f) for f in features)
        
    except Exception:
        # If the file is entirely corrupted or missing, return a tuple of 14 NaNs so the pipeline doesn't break.
        return (np.nan,) * 14


# --- 4. Multiprocessing Worker Function ---
def process_file(row_dict):
    # Isolate the required metadata from the dictionary passed by the multiprocessing pool.
    file_id = row_dict["filename"]
    target = row_dict["target"]
    audio_path = os.path.join(FLAC_DIR, f"{file_id}.flac")
    
    # Only attempt extraction if the physical file exists on the disk.
    if os.path.exists(audio_path):
        feats = measure_pitch_jsh(audio_path)
        # Package the extracted features alongside their core identifiers (filename and label) to maintain data alignment.
        return {
            "filename": file_id, "label": target,
            "meanF0": feats[0], "stdevF0": feats[1], "hnr": feats[2],
            "j_local": feats[3], "j_abs": feats[4], "j_rap": feats[5], "j_ppq5": feats[6], "j_ddp": feats[7],
            "s_local": feats[8], "s_db": feats[9], "s_apq3": feats[10], "s_apq5": feats[11], "s_apq11": feats[12], "s_dda": feats[13]
        }
    # Return None if the file is missing so it can be filtered out later.
    return None


# --- 5. Main Execution ---
def main():
    print("Loading ASVspoof 2019 LA protocol file...")
    # Load the space-separated protocol file into pandas, manually assigning meaningful column names.
    labels_df = pd.read_csv(PROTOCOL_FILE, sep=" ", header=None, names=["speaker_id", "filename", "env", "attack", "label"])
    
    # Convert the string labels ("bonafide" or "spoof") into binary integer targets for downstream model training.
    labels_df["target"] = labels_df["label"].apply(lambda x: 1 if x == "bonafide" else 0)
    
    # Convert the required columns into a list of dictionaries. This format is required to pass data cleanly to the multiprocessing workers.
    tasks = labels_df[["filename", "target"]].to_dict('records')
    
    # Automatically detect the number of available CPU cores to maximize parallel processing efficiency.
    cores = cpu_count()
    
    print(f"Extracting VAD features (Librosa engine) using {cores} CPU cores...")
    # Instantiate the multiprocessing pool, opening one background process per CPU core.
    with Pool(processes=cores) as pool:
        # Use imap to feed the tasks list to the worker function, wrapping it in tqdm to display a progress bar.
        results = list(tqdm(pool.imap(process_file, tasks), total=len(tasks)))
        
    # Filter out any 'None' returns generated by missing audio files.
    features_list = [r for r in results if r is not None]
    
    # Convert the massive list of dictionaries back into a structured pandas DataFrame.
    df = pd.DataFrame(features_list)
    initial_len = len(df)
    
    # Drop any rows where Praat failed completely (indicated by a NaN in the most basic jitter metric).
    df.dropna(subset=['j_local'], inplace=True)
    
    # For rows that successfully extracted most metrics but failed on a few edge cases, mathematically fill the missing spots with the column average.
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    print(f"\\nDropped {initial_len - len(df)} files that were entirely silent or unvoiced.")
    
    # Save the final structured dataset to the hard drive for the downstream model to utilize.
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Academic features saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()