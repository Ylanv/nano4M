# train_raw_ddp.py
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
from audio_tokenizer.vqvae.visualize import stft_loss
from audio_tokenizer.vqvae.models.vqvae import RawVQVAE
from audio_tokenizer.vqvae.data.dataset import LibriSpeechMelDataset

# Hyperparameters
batch_size = 128
learning_rate = 2e-4
epochs = 30
sample_rate = 16000
segment_duration = 2.0
log_every = 50
save_every = 1
embedding_dim = 128
num_embeddings = 512

SAVE_MODEL_PATH = "audio_tokenizer/vqvae/save/"
DATASET_PATH = "/work/com-304/snoupy/librispeech/"
URL = "train-clean-100"
save_folder = Path(SAVE_MODEL_PATH)
save_folder.mkdir(parents=True, exist_ok=True)

def main():
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

    # Data
    dataset = LibriSpeechMelDataset(
        root=Path(DATASET_PATH),
        url=URL,
        segment_duration=segment_duration,
    )
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=1)

    model = RawVQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings).cuda()
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    recon_loss_fn = nn.L1Loss()
    scaler = GradScaler()
    
    for epoch in range(epochs):
        model.train()
        sampler.set_epoch(epoch)
        running_loss = 0.0

        for step, batch in enumerate(tqdm(dataloader, disable=local_rank != 0)):
            waveform, sr, txt = batch
            waveform = waveform.cuda(non_blocking=True)
            waveform = waveform / waveform.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)

            optimizer.zero_grad()

            with autocast():
                x_recon, vq_loss = model(waveform)
                recon_loss = recon_loss_fn(x_recon, waveform)
                loss_sftf = stft_loss(x_recon, waveform)
                loss = recon_loss + vq_loss + loss_sftf

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if step % log_every == 0 and local_rank == 0:
                wandb.log({
                    "loss/total": loss.item(),
                    "loss/recon": recon_loss.item(),
                    "loss/vq": vq_loss.item(),
                    "loss/sftf" : loss_sftf.item(),
                    "epoch": epoch,
                    "step": epoch * len(dataloader) + step,
                })

        if local_rank == 0:
            print(f"Epoch [{epoch+1}] avg loss: {running_loss / len(dataloader):.4f}")
            if (epoch + 1) % save_every == 0:
                torch.save(model.module.state_dict(), save_folder / f"vqvae_epoch{epoch+1}.pt")

    if local_rank == 0:
        wandb.finish()

if __name__ == '__main__':
    main()
