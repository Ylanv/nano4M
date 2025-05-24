import os
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from torch.cuda.amp.autocast_mode import autocast
from pathlib import Path
from tqdm import tqdm

from audio_tokenizer.vqvae.models.vqvae import RawVQVAE
from audio_tokenizer.vqvae.data.audio_utils import stft_loss
from audio_tokenizer.vqvae.data.dataset import LibriSpeechMelDataset

'''
This file evaluate the VQ-VAE on the test set. This can be run using sbatch evaluate.sh
'''
# === Config ===
BATCH_SIZE = 64
SAMPLE_RATE = 16000
SEGMENT_DURATION = 2.0
EMBEDDING_DIM = 128
NUM_EMBEDDINGS = 512
CHECKPOINT_PATH = "/work/com-304/snoupy/weights/vqvae/final/adamw_epoch30.pt"
DATASET_PATH = "/work/com-304/snoupy/librispeech/"
SAVE_RECON = False  # Set to True if you want to save .wav comparisons

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    ''' Load VQ-VAE using CHECKPOINT Weights
    '''
    model = RawVQVAE(embedding_dim=EMBEDDING_DIM, num_embeddings=NUM_EMBEDDINGS).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    return model

@torch.no_grad()
def evaluate(model, dataloader):
    recon_loss_fn = nn.L1Loss()
    total_loss, total_recon, total_vq, total_stft = 0.0, 0.0, 0.0, 0.0

    for i, batch in enumerate(tqdm(dataloader)):
        waveform, sr, _ = batch
        waveform = waveform.to(device)
        waveform = waveform / waveform.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)

        with autocast():
            x_recon, vq_loss = model(waveform)
            recon_loss = recon_loss_fn(x_recon, waveform)
            loss_sftf = stft_loss(x_recon, waveform)
            total = recon_loss + vq_loss + loss_sftf

        total_loss += total.item()
        total_recon += recon_loss.item()
        total_vq += vq_loss.item()
        total_stft += loss_sftf.item()

        # === Save reconstructed sample ===
        if SAVE_RECON and i == 0:
            torchaudio.save("original.wav", waveform[0].cpu(), SAMPLE_RATE)
            torchaudio.save("reconstructed.wav", x_recon[0].cpu(), SAMPLE_RATE)

    N = len(dataloader)
    print(f"\nEvaluation on test-clean:")
    print(f"  Total loss     : {total_loss / N:.4f}")
    print(f"  Recon loss (L1): {total_recon / N:.4f}")
    print(f"  VQ loss        : {total_vq / N:.4f}")
    print(f"  STFT loss      : {total_stft / N:.4f}")

def main():
    model = load_model()

    # === TEST DATASET ===
    test_dataset = LibriSpeechMelDataset(
        root=Path(DATASET_PATH),
        url="test-clean",
        segment_duration=SEGMENT_DURATION,
    )

    # === TEST DATALOADER ===
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    evaluate(model, test_loader)

if __name__ == "__main__":
    main()
