import os
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm


# ---------------- 1. Paths (edit for your setup) ----------------
# Using absolute paths ensures the script can be run from any directory without pathing errors.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Example: ASVspoof 2021 LA train protocol and wav/flac dir
# Change these to the actual ASVspoof 2021 paths on the device
PROTOCOL_FILE = os.path.join(
    BASE_DIR,
    "data", "LA", "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt"
)


AUDIO_DIR = os.path.join(
    BASE_DIR,
    "data", "LA", "ASVspoof2019_LA_train", "flac"
)

# This CSV will act as our lightweight training dataset later, preventing the need to re-process audio.
OUTPUT_CSV = os.path.join(BASE_DIR, "train_ssl_features.csv")

# Wav2Vec2 models are strictly pre-trained on 16kHz audio. Any deviation will corrupt the learned patterns.
SAMPLE_RATE = 16000


# ---------------- 2. Load SSL model ----------------
# Automatically fall back to CPU if a CUDA-enabled GPU isn't available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The processor handles raw audio normalization (scaling amplitudes) before it hits the model.
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)

# Load the base model and immediately push it to the selected device (GPU/CPU).
ssl_model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
).to(device)

# .eval() is critical here. It disables training-specific layers like Dropout, 
# ensuring deterministic outputs and faster execution during feature extraction.
ssl_model.eval()


def load_audio(path, target_sr=SAMPLE_RATE):
    """
    Safely loads and standardizes audio files regardless of their original format.
    """
    # sf.read natively returns a NumPy array, which is ideal for PyTorch conversion.
    wav_numpy, sr = sf.read(path, dtype='float32')
    
    # Convert numpy array to PyTorch tensor for hardware acceleration
    wav = torch.tensor(wav_numpy)
    
    # soundfile returns shape [T] for mono, and [T, Channels] for stereo.
    # Torchaudio resampling strictly expects [Channels, T].
    if wav.ndim == 1:
        # Adds a dummy channel dimension (e.g., shape becomes [1, 50000])
        wav = wav.unsqueeze(0)  
    else:
        # Flips the matrix to fit the [Channels, T] requirement
        wav = wav.t()           
        
    # Dynamically resample if the file isn't 16kHz (prevents Wav2Vec2 pattern corruption)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
        
    # If the audio is stereo (2 channels), mathematically average them into a single mono channel
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        
    # Remove the dummy channel dimension so the processor gets the raw 1D temporal array
    return wav.squeeze(0)


# @torch.no_grad() prevents PyTorch from tracking gradients (the math used for learning).
# This saves massive amounts of VRAM, which is essential for constrained hardware.
@torch.no_grad()
def extract_ssl_embedding(path):
    """
    Passes audio through Wav2Vec2 and temporally pools the hidden states.
    Returns a 1D vector: [mean; std] of the last hidden layer.
    """
    waveform = load_audio(path)  # Shape: [Time]
    
    # Process the audio. We return 'pt' (PyTorch tensors). 
    # No padding is needed here because we are processing batch size 1.
    inputs = processor(
        waveform.numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )
    
    # Move the standardized input tensor to the GPU
    input_values = inputs.input_values.to(device)  # Shape: [1, Length]


    # Pass ONLY input_values to the model (attention masks aren't needed for single files)
    outputs = ssl_model(input_values)
    
    # Extract the deepest contextual representation of the audio
    hidden = outputs.last_hidden_state  # Shape: [Batch=1, Time_Frames, Embedding_Dims]


    # Temporal Pooling: Wav2Vec2 returns features for every fraction of a second.
    # We compress this across the time dimension (dim=1) to get a fixed-size representation 
    # for the entire file, regardless of whether it's 2 seconds or 10 seconds long.
    mean_emb = hidden.mean(dim=1)      # Average features [1, Dims]
    std_emb = hidden.std(dim=1)        # Variance of features [1, Dims]
    
    # Concatenate them side-by-side to capture both the average acoustic profile and its variation
    emb = torch.cat([mean_emb, std_emb], dim=1)  # Shape: [1, 2 * Dims]


    # Aggressive memory management to prevent VRAM bottlenecks on 6GB GPUs
    del inputs, input_values, outputs, hidden
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Move the vector back to system RAM and strip the batch dimension
    return emb.cpu().squeeze(0)  


# ---------------- 3. Read protocol and loop ----------------
def main():
    print("Loading ASVspoof protocol file...")
    
    # 1. Read with sep="\s+" (Regex). The ASVspoof protocol files use inconsistent spacing.
    # This safely treats any sequence of spaces as a single column divider.
    df = pd.read_csv(
        PROTOCOL_FILE, sep="\s+", header=None,
        names=["speaker_id", "filename", "env", "attack", "label"]
    )


    df = df[["filename", "label"]].copy()
    
    # 2. Strip any hidden whitespace or newline characters from the filenames to prevent lookup failures
    df["filename"] = df["filename"].str.strip()
    
    # Convert string classes into binary targets for the downstream neural network
    df["target"] = df["label"].apply(lambda x: 1 if x == "bonafide" else 0)


    feature_rows = []
    print(f"Starting SSL feature extraction for {len(df)} files...")


    # Keep track of missing files so we know immediately if our directory paths are broken
    missing_count = 0


    # tqdm provides a progress bar in the terminal, useful for multi-hour extraction scripts
    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_id = row["filename"]
        target = row["target"]
        
        audio_path = os.path.join(AUDIO_DIR, f"{file_id}.flac")


        # 3. Explicitly check for file existence. 
        if not os.path.exists(audio_path):
            missing_count += 1
            # We only print the first 5 missing files to avoid freezing the terminal with thousands of logs
            if missing_count < 5:  
                print(f"MISSING FILE: Checked path -> {audio_path}")
            continue


        try:
            vec = extract_ssl_embedding(audio_path)  
            
            # Detach from any remaining PyTorch graphs, move to CPU, and convert to standard float32 numpy
            vec_numpy = vec.detach().cpu().to(torch.float32).numpy()
            vec_list = vec_numpy.tolist()
            
            # Initialize a dictionary for this row with its metadata
            feat_dict = {"filename": file_id, "label": target}
            
            # Dynamically map the hundreds of embedding dimensions to named columns (ssl_0, ssl_1, etc.)
            for i, v in enumerate(vec_list):
                feat_dict[f"ssl_{i}"] = float(v)
                
            feature_rows.append(feat_dict)
            
        except Exception as e:
            # Gracefully handle corrupted audio files without crashing the entire multi-hour pipeline
            print(f"\nError for {file_id}: {e}")
            import traceback
            traceback.print_exc()
            continue


    # Failsafe: If every single file was missing, halt before saving an empty CSV
    if missing_count == len(df):
        print("\nCRITICAL: ALL 25,380 FILES ARE MISSING. Check your AUDIO_DIR path!")
        return


    # 4. Save to CSV
    # Convert the list of dictionaries back into a structured tabular format and save to disk
    ssl_df = pd.DataFrame(feature_rows)
    ssl_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Saved SSL features to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()