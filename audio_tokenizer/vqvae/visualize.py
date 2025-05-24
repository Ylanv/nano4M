import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import torchaudio
from wavenet_vocoder.wavenet import WaveNet
from audio_tokenizer.vqvae.models.vqvae import VQVAE,RawVQVAE
from audio_tokenizer.vqvae.data.dataset import LibriSpeechMelDataset
from audio_tokenizer.vqvae.data.audio_utils import visualize_waveform,visualize_spectogram,stft_loss

WEIGHTS_WAVENET = "/work/com-304/snoupy/weights/wavenet/final.pt"
WEIGHTS_VQVAE = "/work/com-304/snoupy/weights/vqvae/final/adamw_epoch13.pt"
DATASET_PATH = "/work/com-304/snoupy/librispeech/"
DATASET = "dev-clean" 
SAVE_PATH = "/work/com-304/snoupy/samples"
ORIGINAL_WAV  = f"{SAVE_PATH}/original.wav" 

# Hyperparameters
batch_size = 1
sample_rate = 16000
segment_duration = 2.0
stride_duration = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
def generate_samples():    
    # === Dataset ===
    # dataset = LibriSpeechMelDataset(
    #     root=Path(DATASET_PATH),
    #     url=DATASET,
    #     segment_duration=segment_duration,
    #     stride_duration=stride_duration,
    # )
    
    # === Dataloader ===
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    #print(f"Dataset size : {len(dataset)}")
    
   
    # === Model ===
    
    # ==== VQ-VAE ==== 
    
    vqvae = RawVQVAE(embedding_dim=128, num_embeddings=512).to(device)
    vqvae.load_state_dict(torch.load(WEIGHTS_VQVAE, weights_only=True))
    vqvae.eval()
    
    # ==== Wavenet ==== 
    # Initialize Wavenet and load weights
    wavenet = WaveNet(
        out_channels=30, #
        layers=20, #
        stacks=2, #
        residual_channels=64, #
        gate_channels=128, # 
        skip_out_channels=64,  # 
        kernel_size=3,# 
        dropout=0.05, # 
        cin_channels=128,  
        scalar_input=True,
        upsample_conditional_features =True,
        upsample_scales=[4,4,4,1]
    ).to(device)
    wavenet.load_state_dict(torch.load(WEIGHTS_WAVENET,weights_only=True))
    wavenet.eval()    

    # === Load first batch === 
    
    waveform, sr = torchaudio.load(ORIGINAL_WAV)
    waveform = waveform.unsqueeze(0).to(device)
        
    print(f"Waveform shape : {waveform.shape}, sample rate {sr}")
    
    # Normalize wf between [-1,1]
    wf_n = waveform = waveform / waveform.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
    
    # === Reconstruct with vqvae ===
    wf_vqvae,_ = vqvae(wf_n)
    
    print(f"Reconstructed wf with VQ-VAE shape : {wf_vqvae.shape}")
    
    # === Reconstruct with wavenet ===
    z_e = vqvae.encode(wf_n)
    print(f"z_e shape : {z_e.shape}")
    with torch.no_grad():
        wf_wavenet = wavenet.incremental_forward(
            initial_input = None,
            c=z_e,
            T=32000,
            softmax=False,
            quantize=False
        )
    
    print(f"Reconstructed wf with wavenet shape : {wf_wavenet.shape}")
    # === Save generate wf ===
    # Detach from CPU
    wf_n = wf_n.squeeze(0).detach().cpu()
    wf_vqvae = wf_vqvae.squeeze(0).detach().cpu()
    wf_wavenet = wf_wavenet.squeeze(0).detach().cpu()
    
    # # Save original
    # torchaudio.save(
    #     ORIGINAL_WAV,
    #     wf_n,
    #     sr,
    #     encoding="PCM_F",
    #     bits_per_sample = 32   
    # )
    
    # Save vqvae 
    torchaudio.save(
        f"{SAVE_PATH}/vqvae.wav",
        wf_vqvae,
        sr,
        encoding="PCM_F",
        bits_per_sample = 32   
    )
    
    # Save wavenet
    torchaudio.save(
        f"{SAVE_PATH}/wavenet.wav",
        wf_wavenet,
        sr,
        encoding="PCM_F",
        bits_per_sample = 32   
    ) 
    
    # Save waveform plot
    visualize_waveform(wf_n,sample_rate,f"{SAVE_PATH}/original.png")
    visualize_waveform(wf_vqvae,sample_rate,f"{SAVE_PATH}/vqvae.png")
    visualize_waveform(wf_wavenet,sample_rate,f"{SAVE_PATH}/wavenet.png")

    # Print reconstruction loss
    l1_loss = nn.L1Loss()
    vqvae_l1 = l1_loss(wf_vqvae,wf_n)
    wavenet_l1 = l1_loss(wf_wavenet,wf_n)
    vqvae_stft = stft_loss(wf_vqvae,wf_n)
    wavenet_stft = stft_loss(wf_wavenet,wf_n)
    
    print(100*"#")
    print(40 * " " + "Loss" + 40 * " ")
    print(100*"#")
    print("VQ-VAE")
    print(f"L1 : {vqvae_l1} | STFT : {vqvae_stft}")
    print("Wavenet")
    print(f"L1 : {wavenet_l1} | STFT : {wavenet_stft}")
    




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
    generate_samples()


if __name__ == "__main__":
    main()
