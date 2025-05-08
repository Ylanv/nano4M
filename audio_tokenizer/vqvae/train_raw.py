import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb

from audio_tokenizer.vqvae.models.vqvae import RawVQVAE
from audio_tokenizer.vqvae.data.dataset import LibriSpeechMelDataset
from torch.cuda.amp import autocast, GradScaler

torch.autograd.set_detect_anomaly(True)

SAVE_MODEL_PATH = "audio_tokenizer/vqvae/save/"
DATASET_PATH = "/work/com-304/snoupy/librispeech/"
URL = "train-clean-100"

save_folder = Path(SAVE_MODEL_PATH)
save_folder.mkdir(parents=True, exist_ok=True)
# Hyperparameters
batch_size = 128
learning_rate = 2e-4
epochs = 10
sample_rate = 16000
segment_duration = 2.0
n_mels = 80
n_fft = 1024
hop_length = 256
log_every = 50
save_every = 1 

# Init WandB
wandb.init(
    entity="scoobyfam",
    project="VQ-VAE",
    config={
        "batch_size": batch_size,
        "lr": learning_rate,
        "epochs": epochs,
        "embedding_dim": 128,
        "num_embeddings": 512,
        "sample_rate": sample_rate,
        "segment_duration": segment_duration,
    },
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
dataset = LibriSpeechMelDataset(
    root=Path(DATASET_PATH),
    url=URL,
    segment_duration=segment_duration,
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

# Model
# Parallelize the Model
model = nn.DataParallel(RawVQVAE(embedding_dim=128, num_embeddings=512)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
recon_loss_fn = nn.L1Loss()

# Training
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for step, batch in enumerate(tqdm(dataloader)):
        waveform, sr, txt = batch
        waveform = waveform.to(device)
        waveform = waveform / waveform.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        x_recon, vq_loss = model(waveform)

    
        # Compute loss
        recon_loss = recon_loss_fn(x_recon, waveform)
        loss = recon_loss + vq_loss
    

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Log losses only
        if step % log_every == 0:
            wandb.log(
                {
                    "loss/total": loss.item(),
                    "loss/recon": recon_loss.item(),
                    "loss/vq": vq_loss.item(),
                    "epoch": epoch,
                    "step": epoch * len(dataloader) + step,
                }
            )

    print(f"Epoch [{epoch+1}] avg loss: {running_loss / len(dataloader):.4f}")
    if(epoch + 1)%save_every == 0:
        pth = save_folder / f"vqvae_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), pth.as_posix())

wandb.finish()
