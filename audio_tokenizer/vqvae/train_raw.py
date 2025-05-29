import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from pathlib import Path
from tqdm import tqdm
import wandb
from audio_tokenizer.vqvae.data.audio_utils import stft_loss
from audio_tokenizer.vqvae.models.vqvae import RawVQVAE
from audio_tokenizer.vqvae.data.dataset import LibriSpeechMelDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split

'''
This file is for training the VQ-VAE using raw waveform from Librispeech.  This can be run using sbatch train.sh 
'''

# === Hyperparameters ===
batch_size = 256
learning_rate = 2e-4
epochs = 30
sample_rate = 16000
segment_duration = 2.0
log_every = 50
save_every = 1
embedding_dim = 128
num_embeddings = 512

# === PATH ===
SAVE_MODEL_PATH = "/work/com-304/snoupy/weights/vqvae/final"
DATASET_PATH = "/work/com-304/snoupy/librispeech/"
URL = "train-clean-360"
save_folder = Path(SAVE_MODEL_PATH)
save_folder.mkdir(parents=True, exist_ok=True)

def main():
    
    # DDP Training
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    # Set up WandB only on rank 0
    if local_rank == 0:
        wandb.init(
            entity="scoobyfam",
            project="VQ-VAE-DDP",
            config={
                "batch_size": batch_size,
                "lr": learning_rate,
                "epochs": epochs,
                "embedding_dim": embedding_dim,
                "num_embeddings": num_embeddings,
                "sample_rate": sample_rate,
                "segment_duration": segment_duration,
            },
        )

    # === DATASET ===
    dataset = LibriSpeechMelDataset(
        root=Path(DATASET_PATH),
        url=URL,
        segment_duration=segment_duration,
    )
    
    # Split train/val 
    val_ratio = 0.1
    total_len = len(dataset)
    val_len = int(val_ratio * total_len)
    train_len = total_len - val_len
    
    train_dataset,val_dataset = random_split(dataset,[train_len,val_len])
    
    # === DATALOADER ===
    # Train dataset
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)

    # Val dataset
    val_sampler = DistributedSampler(val_dataset,shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2, pin_memory=True)  
  
    # === MODEL ===
    model = RawVQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings).cuda()
    model = DDP(model, device_ids=[local_rank])

    # === OPTIMIZER,LOSS AND SCHEDULER ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    recon_loss_fn = nn.L1Loss()
    scaler = GradScaler()
    #scheduler = CosineAnnealingWarmRestarts(optimizer,T_0 = 5*len(train_dataloader),T_mult=1)
    scheduler = CosineAnnealingLR(optimizer,T_max=len(train_dataloader) * epochs)
    
    
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0

        for step, batch in enumerate(tqdm(train_dataloader, disable=local_rank != 0)):

            # Load wf and normalize it between [-1,1]
            waveform, sr, txt = batch
            waveform = waveform.cuda(non_blocking=True)
            waveform = waveform / waveform.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)

            optimizer.zero_grad()

            # Calculate reconstruction loss, stft loss and vq loss.
            with autocast():
                x_recon, vq_loss = model(waveform)
                recon_loss = recon_loss_fn(x_recon, waveform)
                loss_sftf = stft_loss(x_recon, waveform)
                loss = recon_loss + vq_loss + loss_sftf

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()

            # === LOG ===
            if step % log_every == 0 and local_rank == 0:
                wandb.log({
                    "loss/total": loss.item(),
                    "loss/recon": recon_loss.item(),
                    "loss/vq": vq_loss.item(),
                    "loss/sftf" : loss_sftf.item(),
                    "epoch": epoch,
                    "step": epoch * len(train_dataloader) + step,
                    "learning_rate" : scheduler.get_last_lr()[0]
                })
        
        # === VALIDATION EACH EPOCH ===
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for val_batch in val_dataloader:
                waveform, sr, txt = val_batch
                waveform = waveform.cuda(non_blocking=True)
                waveform = waveform / waveform.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)

                optimizer.zero_grad()

                with autocast():
                    x_recon, vq_loss = model(waveform)
                    recon_loss = recon_loss_fn(x_recon, waveform)
                    loss_sftf = stft_loss(x_recon, waveform)
                    loss = recon_loss + vq_loss + loss_sftf
                
                val_loss += loss.item()
                
        # === LOG VAL LOSS AND AVG EPOCH LOSS ===
        val_loss /= len(val_dataloader)
        avg_loss = running_loss / len(train_dataloader)
        if local_rank == 0:
            wandb.log({
                "epoch_loss": avg_loss,
                "epoch": epoch,
                "val_loss":val_loss
            })
            
            # === SAVE WEIGHTS ===
            print(f"Epoch [{epoch+1}] | avg loss: {avg_loss} | val loss : {val_loss}")
            if (epoch + 1) % save_every == 0:
                torch.save(model.module.state_dict(), save_folder / f"adamw_epoch{epoch+1}.pt")

    if local_rank == 0:
        wandb.finish()

if __name__ == '__main__':
    main()
