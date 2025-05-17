import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import torchaudio

from audio_tokenizer.vqvae.models.vqvae import VQVAE,RawVQVAE
from audio_tokenizer.vqvae.data.dataset import LibriSpeechMelDataset
from audio_tokenizer.vqvae.data.audio_utils import (
    Mel_to_wf,
    Wf_to_mel,
    visualize_waveform,
    visualize_spectogram,
    save_waveform_batch
)


SAVE_MODEL_PATH = "audio_tokenizer/vqvae/save/vqvae_epoch30.pt"
DATASET_PATH = "/work/com-304/snoupy/librispeech/"
URL = "dev-clean" #"train-clean-100"

# Hyperparameters
batch_size = 16
sample_rate = 16000
segment_duration = 4.0
stride_duration = 1.0
n_mels = 80
n_fft = 1024
hop_length = 256

def reconstruct():    
    # Dataset
    dataset = LibriSpeechMelDataset(
        root=Path(DATASET_PATH),
        url=URL,
        segment_duration=segment_duration,
        stride_duration=stride_duration,
    )
    # Dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    print(f"Dataset size : {len(dataset)}")
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawVQVAE(embedding_dim=128, num_embeddings=512).to(device)
    model.load_state_dict(torch.load(SAVE_MODEL_PATH, weights_only=True))
    model.eval()
    # Audio utils
    mel_to_wf = Mel_to_wf(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    ).to(device)
    wf_to_mel = Wf_to_mel(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        segment_duration=segment_duration,
    ).to(device)
    loss_fn = nn.L1Loss()
    for batch_idx, batch in enumerate(dataloader):
        waveform, sr, txt = batch
        waveform = waveform.to(device)
        print(50 * "-" + f"Batch {batch_idx}" + 50 * "-")
        print(f"Waveform shape : {waveform.shape}, sample rate {sr}")
        wf_n = waveform = waveform / waveform.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        x_hat,vq_loss = model(wf_n)
        recon_loss = loss_fn(x_hat,wf_n)
        print(f"x hat s:{x_hat.shape}, wf_n : {wf_n.shape}")
        loss_sftf = stft_loss(x_hat, wf_n,n_fft=n_fft,hop_length=hop_length)
        print(f"Recon loss:{recon_loss},vq_loss:{vq_loss},total loss{recon_loss + vq_loss},loss sftf : {loss_sftf}")
        save_waveform_batch(
            wf_n,
            x_hat,
            sample_rate=sample_rate,
            batch_index=batch_idx,
            batch_size=wf_n.shape[0],
            output_dir="audio_tokenizer/vqvae/data/inference",
        )
        #print(f"Transcribe: {txt}")    
        if batch_idx > -1:
            break 
        continue
        mel, mean, std = wf_to_mel(waveform)
        print(
            f"Waveform transformed to mel spectogram, shape : {mel.shape}, mean : {mean}, std : {std}"
        )
        x_recon, vq_loss, _ = model(mel)
        print(f"VQ Loss : {vq_loss}")
        x_recon = x_recon.squeeze(1)
        print(f"Mel spectogram reconstructed, shape : {x_recon.shape}")

        # Match size
        mel_len = mel.shape[-1]
        recon_len = x_recon.shape[-1]
        if recon_len > mel_len:
            x_recon = x_recon[:, :, :mel_len]
        elif recon_len < mel_len:
            x_recon = F.pad(x_recon, (0, mel_len - recon_len), mode="reflect")

        print(f"Cropping/Padding, shape : {x_recon.shape}")
        x_recon = x_recon * (std + 1e-6) + mean
        recon_loss_fn = nn.MSELoss()
        loss = recon_loss_fn(mel, x_recon)
        print(f"Reconstruction Loss : {loss}")
        waveform_recon = mel_to_wf(x_recon)
        print(f"Mel spectogram transformed to waveform, shape : {waveform_recon.shape}")
        save_waveform_batch(
            waveform,
            waveform_recon,
            sample_rate=sample_rate,
            batch_index=batch_idx,
            batch_size=waveform.shape[0],
            output_dir="audio_tokenizer/vqvae/data/inference",
        )
        print(100 * "-")
        if batch_idx > -1:
            break





def check_codebook_similarity(embedding: torch.nn.Embedding, threshold=1e-3):
    """
    Checks how similar the embeddings in a codebook are.
    Args:
        embedding (nn.Embedding): your codebook
        threshold (float): tolerance to consider two embeddings as "equal"
    """
    weights = embedding.weight.detach().cpu()  # (num_embeddings, embedding_dim)
    print(f"Codebook shape : {weights.shape}")
    num_embeddings = weights.size(0)

    # Compute pairwise L2 distance matrix
    dist_matrix = torch.cdist(weights, weights, p=2)  # (N, N)

    # Set diagonal to large value to ignore self-distance
    dist_matrix.fill_diagonal_(float("inf"))
    print(dist_matrix)
    # Count how many are nearly identical
    num_identical = (dist_matrix < threshold).sum().item()
    print(f"Total nearly-identical embeddings (< {threshold}): {num_identical}")

    # Optional: print min, mean, and max distances
    print(f"Min distance: {dist_matrix.min().item():.6f}")
    print(f"Mean distance: {dist_matrix.mean().item():.6f}")
    print(f"Max distance: {dist_matrix.max().item():.6f}")



def main():
    reconstruct()


if __name__ == "__main__":
    main()
