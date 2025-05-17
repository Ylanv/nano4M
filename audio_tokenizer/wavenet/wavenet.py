from wavenet_vocoder.wavenet import WaveNet
from audio_tokenizer.vqvae.models.vqvae import RawVQVAE
import torch 
import torchaudio
import os 

MODEL_PATH_VQVAE = "/work/com-304/snoupy/weights/vqvae/vqvae_epoch30.pt"
MODEL_PATH_WAVENET = "/work/com-304/snoupy/weights/wavenet/wavenet_epoch_3.pt"
SAMPLE_WAV_PATH = "audio_tokenizer/vqvae/data/inference/batch_0/original_0.wav"
SAVE_OUTPUT = "audio_tokenizer/vqvae/data/inference/generated.wav"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    
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
        cin_channels=128,  # This must match D in your [B, T, D]
        scalar_input=True,  # True if you're using [-1,1] float waveform, False for mu-law
        upsample_conditional_features =True,
        upsample_scales=[4,4,4,1]
    ).to(device)
    wavenet.load_state_dict(torch.load(MODEL_PATH_WAVENET,weights_only=True))
    wavenet.eval()
    
    # Initialize VQVAE and load weights
    vqvae = RawVQVAE().to(device)
    vqvae.load_state_dict(torch.load(MODEL_PATH_VQVAE, weights_only=True))
    vqvae.eval()
    
    # Random waveform of 2S at 16khz 
    x,sr = torchaudio.load(uri =SAMPLE_WAV_PATH,num_frames=32000)  # shape: [B, 1, T], float values in [-1, 1]
    x = x.unsqueeze(0).to(device)
    x = x / x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
    
    print(f"x shape : {x.shape},sr:{sr}")
    z_e = vqvae.encode(x)
    print(f"z_e shape : {z_e.shape}")
    c = z_e
    print(f"C shape : {c.shape}")
    
    with torch.no_grad():
    
        generated_wf = wavenet.incremental_forward(
            initial_input = None,
            c=c,
            T=32000,
            softmax=False,
            quantize=False
        )
        
        print(f"Output shape: {generated_wf.shape}")   
        # Save generated_wf 
        wf = generated_wf.squeeze(0).detach().cpu()
        torchaudio.save(
            SAVE_OUTPUT,
            wf,
            sr,
            encoding="PCM_F",
            bits_per_sample = 32   
        )