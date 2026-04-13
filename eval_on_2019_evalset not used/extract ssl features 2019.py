import os
import pandas as pd
import numpy as np
import torch
import torchaudio
import soundfile as sf
import glob
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

# ---------------- 1. Paths ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Point to the 2019 LA Evaluation Protocol and FLACs
PROTOCOL_FILE = os.path.join(BASE_DIR, "data", "LA", "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.eval.trl.txt")
FLAC_DIR = os.path.join(BASE_DIR, "data", "LA", "ASVspoof2019_LA_eval", "flac")
OUTPUT_CSV = os.path.join(BASE_DIR, "eval_la_ssl_features.csv")

SAMPLE_RATE = 16000
BATCH_SIZE = 16  
SAVE_INTERVAL = 10000  # Lowered slightly since LA eval is ~71k files total

# ---------------- 2. PyTorch Dataset ----------------
class ASVspoofDataset(Dataset):
    def __init__(self, dataframe, audio_dir):
        self.df = dataframe
        self.audio_dir = audio_dir
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = row['filename']
        target = row['target']
        codec = row['codec']
        
        audio_path = os.path.join(self.audio_dir, f"{file_id}.flac")
        
        try:
            wav_numpy, sr = sf.read(audio_path, dtype='float32')
            wav = torch.tensor(wav_numpy)
            
            if wav.ndim == 1: wav = wav.unsqueeze(0)
            else: wav = wav.t()
                
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=SAMPLE_RATE)
                
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
                
            waveform = wav.squeeze(0)
        except Exception:
            waveform = torch.zeros(SAMPLE_RATE)
            file_id = f"ERROR_{file_id}"

        return waveform, file_id, target, codec

def pad_collate(batch):
    waveforms, file_ids, targets, codecs = zip(*batch)
    max_len = max([w.shape[0] for w in waveforms])
    padded_wavs = [torch.nn.functional.pad(w, (0, max_len - w.shape[0])) for w in waveforms]
    return torch.stack(padded_wavs), file_ids, targets, codecs

# ---------------- 3. Main Execution ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading Wav2Vec2 model to GPU...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    ssl_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    ssl_model.eval()

    print("Loading 2019 LA Eval Protocol file...")
    # LA Protocol is 5 columns, space-separated
    labels_df = pd.read_csv(PROTOCOL_FILE, sep=" ", header=None, names=["speaker_id", "filename", "env", "attack", "label"])
    labels_df["target"] = labels_df["label"].apply(lambda x: 1 if x == "bonafide" else 0)
    labels_df["codec"] = "nocodec" # Dummy codec to match DF extraction format
    labels_df = labels_df[["filename", "target", "codec"]]

    # --- RESUME LOGIC ---
    processed_filenames = set()
    ckpt_files = sorted(glob.glob(OUTPUT_CSV.replace(".csv", "_ckpt*.csv")))
    
    for f in ckpt_files:
        df_ckpt = pd.read_csv(f, usecols=["filename"])
        processed_filenames.update(df_ckpt["filename"].tolist())
        
    if len(processed_filenames) > 0:
        print(f"\n[Resume] Found {len(processed_filenames)} already processed files in checkpoints.")
        labels_df = labels_df[~labels_df["filename"].isin(processed_filenames)].reset_index(drop=True)
        print(f"[Resume] Remaining files to process: {len(labels_df)}\n")
    
    if len(labels_df) == 0:
        print("All files have already been processed!")
        merge_checkpoints(ckpt_files)
        return

    checkpoint_num = len(ckpt_files) + 1

    dataset = ASVspoofDataset(labels_df, FLAC_DIR)
    # Using 8 workers to speed up audio loading on your LOQ
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate, num_workers=8, pin_memory=True)
    
    feature_rows = []
    
    print(f"Starting batched SSL extraction (Batch Size: {BATCH_SIZE})...")
    with torch.no_grad():
        for waveforms, file_ids, targets, codecs in tqdm(loader, desc="Processing Batches"):
            input_values = waveforms.to(device)
            outputs = ssl_model(input_values)
            hidden = outputs.last_hidden_state  
            
            mean_emb = hidden.mean(dim=1)       
            std_emb = hidden.std(dim=1)         
            batch_emb = torch.cat([mean_emb, std_emb], dim=1).cpu().numpy()
            
            for i in range(len(file_ids)):
                f_id = file_ids[i]
                if f_id.startswith("ERROR_"): continue
                    
                feat_dict = {"filename": f_id, "codec": codecs[i], "label": targets[i]}
                for j, val in enumerate(batch_emb[i].tolist()):
                    feat_dict[f"ssl_{j}"] = float(val)
                feature_rows.append(feat_dict)
                
            # --- RAM SAFE CHECKPOINT SAVING ---
            if len(feature_rows) >= SAVE_INTERVAL:
                temp_csv = OUTPUT_CSV.replace(".csv", f"_ckpt{checkpoint_num}.csv")
                pd.DataFrame(feature_rows).to_csv(temp_csv, index=False)
                print(f"\n[Checkpoint] Saved chunk of {len(feature_rows)} files to {temp_csv}")
                feature_rows = []
                checkpoint_num += 1

    if len(feature_rows) > 0:
        temp_csv = OUTPUT_CSV.replace(".csv", f"_ckpt{checkpoint_num}.csv")
        pd.DataFrame(feature_rows).to_csv(temp_csv, index=False)
        print(f"\n[Checkpoint] Saved final chunk of {len(feature_rows)} files to {temp_csv}")

    all_ckpt_files = sorted(glob.glob(OUTPUT_CSV.replace(".csv", "_ckpt*.csv")))
    merge_checkpoints(all_ckpt_files)

def merge_checkpoints(ckpt_files):
    print("\nMerging all checkpoints into final dataset...")
    first = True
    for f in tqdm(ckpt_files, desc="Merging"):
        chunk = pd.read_csv(f)
        chunk.to_csv(OUTPUT_CSV, mode='w' if first else 'a', header=first, index=False)
        first = False
    print(f"Done! Final CSV successfully saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()