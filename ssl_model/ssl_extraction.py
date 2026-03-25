import os
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

# ---------------- 1. Paths (edit for your setup) ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Example: ASVspoof 2021 LA train protocol and wav/flac dir
# Change these to the actual ASVspoof 2021 paths on the device
PROTOCOL_FILE = os.path.join(
    BASE_DIR,
    "data", "ASVspoof2021", "LA", "ASVspoof2021.LA.cm.train.trn.txt"
)

AUDIO_DIR = os.path.join(
    BASE_DIR,
    "data", "ASVspoof2021", "LA", "ASVspoof2021_LA_train", "flac"
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
    wav, sr = torchaudio.load(path)  # [C, T]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    # mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0)  # [T]


@torch.no_grad()
def extract_ssl_embedding(path):
    """
    Returns a 2D-dimensional vector: [mean; std] of last hidden layer.
    """
    waveform = load_audio(path)  # [T]
    inputs = processor(
        waveform.numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )
    input_values = inputs.input_values.to(device)       # [1, L]
    attention_mask = inputs.attention_mask.to(device)   # [1, L]

    outputs = ssl_model(input_values, attention_mask=attention_mask)
    hidden = outputs.last_hidden_state  # [1, T', D]

    mean_emb = hidden.mean(dim=1)      # [1, D]
    std_emb = hidden.std(dim=1)        # [1, D]
    emb = torch.cat([mean_emb, std_emb], dim=1)  # [1, 2D]

    return emb.cpu().squeeze(0)  # [2D]


# ---------------- 3. Read protocol and loop ----------------
def main():
    print("Loading ASVspoof protocol file...")

    # Adjust columns if needed depending on the exact protocol format
    # For ASVspoof 2021 LA: speaker, filename, system, key
    # or: speaker_id | filename | env | attack | key
    df = pd.read_csv(
        PROTOCOL_FILE, sep=" ", header=None
    )

    # Try to detect format: last column is key (bonafide/spoof)
    # filename is usually at index 1
    df = df[[1, df.columns[-1]]]
    df.columns = ["filename", "label"]

    # bonafide -> 1, spoof -> 0
    df["target"] = df["label"].apply(lambda x: 1 if x == "bonafide" else 0)

    feature_rows = []

    print(f"Starting SSL feature extraction for {len(df)} files...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_id = row["filename"]
        target = row["target"]

        audio_path = os.path.join(AUDIO_DIR, f"{file_id}.flac")
        if not os.path.exists(audio_path):
            # fallback to .wav if needed
            alt_path = os.path.join(AUDIO_DIR, f"{file_id}.wav")
            if os.path.exists(alt_path):
                audio_path = alt_path
            else:
                print(f"Warning: file not found: {file_id}")
                continue

        try:
            vec = extract_ssl_embedding(audio_path)  # [2D]
            feat_dict = {"filename": file_id, "label": target}
            # expand to named columns ssl_0, ssl_1, ...
            for i, v in enumerate(vec.numpy()):
                feat_dict[f"ssl_{i}"] = float(v)
            feature_rows.append(feat_dict)
        except Exception as e:
            print(f"Error for {file_id}: {e}")
            continue

    ssl_df = pd.DataFrame(feature_rows)
    ssl_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Saved SSL features to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
