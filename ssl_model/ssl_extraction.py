import os
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

# ---------------- 1. Paths (edit for your setup) ----------------
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

OUTPUT_CSV = os.path.join(BASE_DIR, "train_ssl_features.csv")

SAMPLE_RATE = 16000

# ---------------- 2. Load SSL model ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)
ssl_model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
).to(device)
ssl_model.eval()


def load_audio(path, target_sr=SAMPLE_RATE):
    wav_numpy, sr = sf.read(path, dtype='float32')
    
    # Convert numpy array to PyTorch tensor
    wav = torch.tensor(wav_numpy)
    
    # soundfile returns shape [T] for mono, and [T, Channels] for stereo.
    # We need it to be [Channels, T] for torchaudio resampling.
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)  # Make it [1, T]
    else:
        wav = wav.t()           # Transpose to [C, T]
        
    # Resample if the sample rate doesn't match 16000
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
        
    # Convert to mono if it has multiple channels
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        
    return wav.squeeze(0)

@torch.no_grad()
def extract_ssl_embedding(path):
    """
    Returns a 2D-dimensional vector: [mean; std] of last hidden layer.
    """
    waveform = load_audio(path)  # [T]
    
    # Process the audio. No padding needed for batch size 1.
    inputs = processor(
        waveform.numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )
    
    input_values = inputs.input_values.to(device)  # [1, L]

    # Pass ONLY input_values to the model (no attention mask)
    outputs = ssl_model(input_values)
    hidden = outputs.last_hidden_state  # [1, T', D]

    mean_emb = hidden.mean(dim=1)      # [1, D]
    std_emb = hidden.std(dim=1)        # [1, D]
    emb = torch.cat([mean_emb, std_emb], dim=1)  # [1, 2D]

    # Clean up to prevent GPU out-of-memory errors
    del inputs, input_values, outputs, hidden
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return emb.cpu().squeeze(0)  # [2D]

# ---------------- 3. Read protocol and loop ----------------
def main():
    print("Loading ASVspoof protocol file...")
    
    # 1. Read with sep="\s+" to handle any number of spaces perfectly
    df = pd.read_csv(
        PROTOCOL_FILE, sep="\s+", header=None,
        names=["speaker_id", "filename", "env", "attack", "label"]
    )

    df = df[["filename", "label"]].copy()
    
    # 2. Strip any hidden whitespace from the filenames
    df["filename"] = df["filename"].str.strip()
    
    df["target"] = df["label"].apply(lambda x: 1 if x == "bonafide" else 0)

    feature_rows = []
    print(f"Starting SSL feature extraction for {len(df)} files...")

    # Keep track of missing files so we know if paths are broken
    missing_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_id = row["filename"]
        target = row["target"]
        
        audio_path = os.path.join(AUDIO_DIR, f"{file_id}.flac")

        # 3. Explicitly print a warning instead of silently continuing
        if not os.path.exists(audio_path):
            missing_count += 1
            if missing_count < 5:  # Only print first few to avoid spamming terminal
                print(f"MISSING FILE: Checked path -> {audio_path}")
            continue

        try:
            vec = extract_ssl_embedding(audio_path)  
            vec_numpy = vec.detach().cpu().to(torch.float32).numpy()
            vec_list = vec_numpy.tolist()
            
            feat_dict = {"filename": file_id, "label": target}
            for i, v in enumerate(vec_list):
                feat_dict[f"ssl_{i}"] = float(v)
                
            feature_rows.append(feat_dict)
            
        except Exception as e:
            print(f"\nError for {file_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if missing_count == len(df):
        print("\nCRITICAL: ALL 25,380 FILES ARE MISSING. Check your AUDIO_DIR path!")
        return

    # Save to CSV
    ssl_df = pd.DataFrame(feature_rows)
    ssl_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Saved SSL features to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()