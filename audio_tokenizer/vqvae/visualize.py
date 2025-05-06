import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path


from audio_tokenizer.vqvae.models.vqvae import VQVAE
from audio_tokenizer.vqvae.data.dataset import LibriSpeechMelDataset
from audio_tokenizer.vqvae.data.audio_utils import (
    Mel_to_wf,
    Wf_to_mel,
    visualize_waveform,
    visualize_spectogram,
)
import torchaudio


def reconstruct():

    # Hyperparameters
    batch_size = 2
    sample_rate = 16000
    segment_duration = 5.0
    n_mels = 80
    n_fft = 1024
    hop_length = 256

    # Dataset
    dataset = LibriSpeechMelDataset(
        root=Path("audio_tokenizer/vqvae/data"), url="train-clean-100"
    )
    # Dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAE(embedding_dim=64, num_embeddings=512).to(device)
    model.load_state_dict(
        torch.load("audio_tokenizer/vqvae/vqvae_final-1.pt", weights_only=True)
    )
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

    for batch_idx, batch in enumerate(dataloader):
        waveform, sr, txt = batch
        waveform = waveform.to(device)
        print(50 * "-" + f"Batch {batch_idx}" + 50 * "-")
        print(f"Waveform shape : {waveform.shape}, sample rate {sr}")
        print(f"Transcribe: {txt}")

        mel, mean, std = wf_to_mel(waveform)
        print(
            f"Waveform transformed to mel spectogram, shape : {mel.shape}, mean : {mean}, std : {std}"
        )
        x_recon, vq_loss, _ = model(mel)
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


def save_waveform_batch(
    originals,
    reconstructions,
    sample_rate,
    batch_index,
    batch_size,
    output_dir="audio_tokenizer/vqvae/data/inference",
):
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

    # Detach tensor and put on CPU
    orig = originals.detach().cpu()
    recon = reconstructions.detach().cpu()

    for i in range(batch_size):
        # Extract wf from batch
        recon_wf_i = recon[i].unsqueeze(0)
        orig_wf_i = orig[i]
        # Create path
        orig_path = batch_folder / f"original_{i}.wav"
        recon_path = batch_folder / f"reconstruction_{i}.wav"
        orig_wf_path = batch_folder / f"original_{i}.png"
        recon_wf_path = batch_folder / f"reconstruction_{i}.png"
        # Save
        torchaudio.save(
            orig_path.as_posix(),
            orig_wf_i,
            sample_rate,
            encoding="PCM_F",
            bits_per_sample=32,
        )
        torchaudio.save(
            recon_path.as_posix(),
            recon_wf_i,
            sample_rate,
            encoding="PCM_F",
            bits_per_sample=32,
        )
        visualize_waveform(orig_wf_i, sample_rate, orig_wf_path.as_posix())
        visualize_waveform(recon_wf_i, sample_rate, recon_wf_path.as_posix())
        print(f"Saved item {i} of batch {batch_index} at {orig_path}")


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
