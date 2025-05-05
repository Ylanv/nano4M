import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb
from audio_tokenizer.vqvae.models.vqvae import VQVAE
from audio_tokenizer.vqvae.data.dataset import LibriSpeechMelDataset
from audio_tokenizer.vqvae.data.audio_utils import MelToAudio
import torchaudio 

def reconstruct():
    # Data
    # Hyperparameters
    batch_size = 16
    learning_rate = 3e-4
    epochs = 10
    sample_rate = 44100
    segment_duration = 2.0
    log_every = 50
    dataset = LibriSpeechMelDataset(
        root=Path("audio_tokenizer/vqvae/data"),
        url = "dev-clean",
        sr=sample_rate,
        segment_duration=segment_duration
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAE(embedding_dim=64, num_embeddings=512).to(device)
    model.load_state_dict(torch.load("audio_tokenizer/vqvae/vqvae_final.pt",weights_only = True))
    model.eval() 
    melToAudio = MelToAudio(sample_rate=sample_rate)
    
    for batch_idx,batch in enumerate(dataloader):
        mel = batch.to(device)  # [B, 80, T]
        x_recon, vq_loss, _ = model(mel)
        x_recon = x_recon.squeeze(1)  # [B, 80, T']

        # Match size
        mel_len = mel.shape[-1]
        recon_len = x_recon.shape[-1]
        if recon_len > mel_len:
            x_recon = x_recon[:, :, :mel_len]
        elif recon_len < mel_len:
            x_recon = F.pad(x_recon, (0, mel_len - recon_len), mode="reflect")
        waveform = melToAudio(x_recon)
        save_waveform_batch(mel.cpu(),waveform.cpu(),sample_rate,batch_idx)
        break


def save_waveform_batch(originals, reconstructions, sample_rate, batch_index, output_dir="audio_tokenizer/vqvae/data/inference"):
    """
    Saves original and reconstructed waveforms for each item in the batch.

    Args:
        originals (Tensor): shape (batch_size, num_channels, num_samples)
        reconstructions (Tensor): shape (batch_size, num_channels, num_samples)
        sample_rate (int): sampling rate (e.g., 16000)
        batch_index (int): index of the batch
        output_dir (str or Path): root folder for output
    """
    batch_folder = Path(output_dir) / f"batch_{batch_index}"
    batch_folder.mkdir(parents=True, exist_ok=True)

    for i, (orig, recon) in enumerate(zip(originals, reconstructions)):
        orig_path = batch_folder / f"original_{i}.wav"
        recon_path = batch_folder / f"reconstruction_{i}.wav"

        torchaudio.save(orig_path.as_posix(), orig, sample_rate)
        torchaudio.save(recon_path.as_posix(), recon, sample_rate)

def main():
    reconstruct()
    
if __name__ == "__main__":
    main()