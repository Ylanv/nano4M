import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb

from audio_tokenizer.vqvae.models.vqvae import VQVAE
from audio_tokenizer.vqvae.data.dataset import LibriSpeechMelDataset

# Hyperparameters
batch_size = 16
learning_rate = 3e-4
epochs = 10
sample_rate = 44100
segment_duration = 2.0
log_every = 50

# Init WandB
wandb.init(
    entity="scoobyfam",
    project="VQ-VAE",
    config={
        "batch_size": batch_size,
        "lr": learning_rate,
        "epochs": epochs,
        "embedding_dim": 64,
        "num_embeddings": 512,
        "sample_rate": sample_rate,
        "segment_duration": segment_duration,
    }
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
dataset = LibriSpeechMelDataset(
    root=Path("audio_tokenizer/vqvae/data"),
    url = "train-clean-100",
    sr=sample_rate,
    segment_duration=segment_duration
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

# Model
model = VQVAE(embedding_dim=64, num_embeddings=512).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
recon_loss_fn = nn.MSELoss()

# Training
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for step, mel in enumerate(tqdm(dataloader)):
        mel = mel.to(device)  # [B, 80, T]
        x_recon, vq_loss, _ = model(mel)
        x_recon = x_recon.squeeze(1)  # [B, 80, T']

        # Match size
        mel_len = mel.shape[-1]
        recon_len = x_recon.shape[-1]
        if recon_len > mel_len:
            x_recon = x_recon[:, :, :mel_len]
        elif recon_len < mel_len:
            x_recon = F.pad(x_recon, (0, mel_len - recon_len), mode="reflect")

        # Compute loss
        recon_loss = recon_loss_fn(x_recon, mel)
        loss = recon_loss + vq_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Log losses only
        if step % log_every == 0:
            wandb.log({
                "loss/total": loss.item(),
                "loss/recon": recon_loss.item(),
                "loss/vq": vq_loss.item(),
                "epoch": epoch,
                "step": epoch * len(dataloader) + step
            })

    print(f"Epoch [{epoch+1}] avg loss: {running_loss / len(dataloader):.4f}")
    #torch.save(model.state_dict(), f"vqvae_epoch{epoch+1}.pt")
torch.save(model.state_dict(), "vqvae_final.pt")
wandb.finish()
