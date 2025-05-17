import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
from pathlib import Path

from wavenet_vocoder.wavenet import WaveNet
from wavenet_vocoder.mixture import discretized_mix_logistic_loss
from audio_tokenizer.vqvae.models.vqvae import RawVQVAE
from audio_tokenizer.vqvae.data.dataset import LibriSpeechMelDataset

# Paths & Constants
LOAD_MODEL_PATH = "/work/com-304/snoupy/weights/vqvae/vqvae_epoch30.pt"
SAVE_MODEL_PATH = Path("/work/com-304/snoupy/weights/wavenet")
DATASET_PATH = Path("/work/com-304/snoupy/librispeech/")
URL = "train-clean-100"

# Hyperparameters
batch_size = 64
learning_rate = 2e-4
epochs = 30
segment_duration = 2.0
embedding_dim = 128
num_embeddings = 512
log_every = 10
save_every = 1


def main():
    # === DDP Setup ===
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)
    global_rank = dist.get_rank()

    # === WandB Setup ===
    if global_rank == 0:
        wandb.init(project="wavenet-vqvae", name="train-wavenet-ddp", config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "embedding_dim": embedding_dim,
            "num_embeddings": num_embeddings
        })

    # === Data ===
    dataset = LibriSpeechMelDataset(root=DATASET_PATH, url=URL, segment_duration=segment_duration)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)

    # === VQ-VAE Encoder (frozen) ===
    vqvae = RawVQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings).to(device)
    vqvae.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location=device))
    vqvae.eval()

    # === WaveNet ===
    wavenet = WaveNet(
        out_channels=30,
        layers=20,
        stacks=2,
        residual_channels=64,
        gate_channels=128,
        skip_out_channels=64,
        kernel_size=3,
        dropout=0.05,
        cin_channels=embedding_dim,
        scalar_input=True,
        upsample_conditional_features=True,
        upsample_scales=[4, 4, 4, 1]
    ).to(device)
    wavenet = DDP(wavenet, device_ids=[local_rank],find_unused_parameters=True)

    optimizer = torch.optim.Adam(wavenet.parameters(), lr=learning_rate)
    scaler = GradScaler()

    # === Training Loop ===
    for epoch in range(1, epochs + 1):
        sampler.set_epoch(epoch)
        wavenet.train()
        total_loss = 0.0

        for step, batch in enumerate(tqdm(dataloader, disable=global_rank != 0, desc=f"Epoch {epoch}")):
            waveform, _, _ = batch
            waveform = waveform.to(device, non_blocking=True)
            waveform = waveform / waveform.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)

            with torch.no_grad():
                z_q = vqvae.encode(waveform)

            optimizer.zero_grad()
            with autocast():
                y_hat = wavenet(waveform, c=z_q)
                loss = discretized_mix_logistic_loss(y_hat, waveform.transpose(1, 2), log_scale_min=-7.0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if step % log_every == 0 and global_rank == 0:
                wandb.log({
                    "step_loss": loss.item(),
                    "epoch": epoch,
                    "step": step + epoch * len(dataloader)
                })

        avg_loss = total_loss / len(dataloader)
        if global_rank == 0:
            wandb.log({"epoch_loss": avg_loss, "epoch": epoch})
            print(f"[Rank 0] Epoch {epoch} | Avg Loss = {avg_loss:.4f}")

            if epoch % save_every == 0:
                SAVE_MODEL_PATH.mkdir(parents=True, exist_ok=True)
                torch.save(wavenet.module.state_dict(), SAVE_MODEL_PATH / f"wavenet_epoch_{epoch}.pt")

    if global_rank == 0:
        wandb.finish()


if __name__ == '__main__':
    main()
