import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import wandb
from pathlib import Path

from wavenet_vocoder.wavenet import WaveNet
from wavenet_vocoder.mixture import discretized_mix_logistic_loss
from audio_tokenizer.vqvae.models.vqvae import RawVQVAE
from audio_tokenizer.vqvae.data.dataset import LibriSpeechMelDataset
from torch.utils.data import random_split

'''
File used for training wavenet conditioned on VQ-VAE latent. Training can be run using sbatch train_wavenet.sh
'''

# === Paths & Constants ===
LOAD_MODEL_PATH = "/work/com-304/snoupy/weights/vqvae/final/final.pt"
SAVE_MODEL_PATH = Path("/work/com-304/snoupy/weights/wavenet")
DATASET_PATH = Path("/work/com-304/snoupy/librispeech/")
URL = "train-clean-360"

# === Hyperparameters ===
batch_size = 64
learning_rate = 1e-4
epochs = 30
segment_duration = 2.0
embedding_dim = 128
num_embeddings = 512
log_every = 10
save_every = 1


def main():
    
    # Print number of GPU 
    print("GPUs visible to this process:", torch.cuda.device_count())
    
    # === DDP Setup ===
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)
    global_rank = dist.get_rank()

    # === WandB Setup ===
    if global_rank == 0:
        wandb.init(
            entity="scoobyfam",
            project="wavenet-vqvae", 
            name="train-wavenet-ddp", 
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "embedding_dim": embedding_dim,
                "num_embeddings": num_embeddings
            }
        )

    # === Data ===
    # Dataset
    dataset = LibriSpeechMelDataset(root=DATASET_PATH, url=URL, segment_duration=segment_duration)
    
    # Split train/val 
    val_ratio = 0.1
    total_len = len(dataset)
    val_len = int(val_ratio * total_len)
    train_len = total_len - val_len
    
    train_dataset,val_dataset = random_split(dataset,[train_len,val_len])
    
    # Train dataset
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)

    # Val dataset
    val_sampler = DistributedSampler(val_dataset,shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2, pin_memory=True)
    
    # === VQ-VAE Encoder in EVAL mode ===
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
    wavenet = DDP(wavenet, device_ids=[local_rank], find_unused_parameters=True)

    # === OPTIMIZER & SCHEDULER ===
    optimizer = torch.optim.Adam(wavenet.parameters(), lr=learning_rate)
    scaler = GradScaler()
    # Warm restart 5 epoch to avoid stagnation
    scheduler = CosineAnnealingWarmRestarts(optimizer,T_0 = 5*len(train_dataloader),T_mult=1)

    # === Training Loop ===
    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)
        wavenet.train()
        total_loss = 0.0

        for step, batch in enumerate(tqdm(train_dataloader, disable=global_rank != 0, desc=f"Epoch {epoch}")):
            # === LOAD WF and Normalize between [-1,1] === 
            waveform, _, _ = batch
            waveform = waveform.to(device, non_blocking=True)
            waveform = waveform / waveform.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)

            # Get VQ-VAE Latent i.e Z_q(x)
            with torch.no_grad():
                z_q = vqvae.encode(waveform)

            # Use VQ-VAE Latent to generate discrete mixture of logistic distribution
            optimizer.zero_grad()
            with autocast():
                y_hat = wavenet(waveform, c=z_q)
                loss = discretized_mix_logistic_loss(y_hat, waveform.transpose(1, 2))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Unscale before clipping
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

            if step % log_every == 0 and global_rank == 0:
                # Log gradient norm
                total_norm = 0.0
                for p in wavenet.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                # === LOG ===
                wandb.log({
                    "step_loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": total_norm,
                    "epoch": epoch,
                    "step": step + epoch * len(train_dataloader)
                })
            
            # Clip gradient 
            clip_grad_norm_(wavenet.parameters(), max_norm=1.0)

        # === Validation Loss ===  
        val_loss = 0.0
        wavenet.eval()
        with torch.no_grad():
            for val_batch in val_dataloader:
                waveform, _, _ = val_batch
                waveform = waveform.to(device, non_blocking=True)
                waveform = waveform / waveform.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)

                z_q = vqvae.encode(waveform)
                y_hat = wavenet(waveform, c=z_q)
                loss = discretized_mix_logistic_loss(y_hat, waveform.transpose(1, 2), log_scale_min=-7.0)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        avg_loss = total_loss / len(train_dataloader)
        if global_rank == 0:
            # Log epoch avg loss and val loss
            wandb.log({
                "epoch_loss": avg_loss,
                "epoch": epoch,
                "val_loss":val_loss
            })
            
            print(f"[Rank 0] Epoch {epoch} | Avg Loss = {avg_loss:.4f} | Val loss = {val_loss}")

            # === SAVE WEIGHTS ===
            if epoch % save_every == 0:
                SAVE_MODEL_PATH.mkdir(parents=True, exist_ok=True)
                torch.save(wavenet.module.state_dict(), SAVE_MODEL_PATH / f"wavenet_epoch_{epoch}.pt")

        
    if global_rank == 0:
        wandb.finish()


if __name__ == '__main__':
    main()
