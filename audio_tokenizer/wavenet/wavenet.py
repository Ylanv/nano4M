from wavenet_vocoder.wavenet import WaveNet
from audio_tokenizer.vqvae.models.vqvae import RawVQVAE
import torch 
import torchaudio
import os 
import torch.nn.functional as F
from wavenet_vocoder.mixture import discretized_mix_logistic_loss
'''
Wavenet testing file, showcase of to train and infer. 
'''

MODEL_PATH_VQVAE = "/work/com-304/snoupy/weights/vqvae/vqvae_epoch30.pt"
MODEL_PATH_WAVENET = "/work/com-304/snoupy/weights/wavenet/wavenet_epoch_3.pt"
SAMPLE_WAV_PATH = "audio_tokenizer/vqvae/data/inference/batch_0/original_0.wav"
SAVE_OUTPUT = "audio_tokenizer/vqvae/data/inference/generated.wav"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
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
        cin_channels=128,
        scalar_input=True,
        upsample_conditional_features=True,
        upsample_scales=[4, 4, 4, 1]
    ).to(device)
    
    waveform = torch.rand(16,1,32000).to(device)[:,:,:-1]
    
    z_q = RawVQVAE().to(device).encode(waveform)
    
    print(f"Latent shape : {z_q.shape}")
    
    y_hat = wavenet(waveform,c=z_q)
    print(f"Reconstruction shape : {y_hat.shape}")
    
    y = waveform[:,:,0:].transpose(1,2)
    print(f"y shape : {y.shape}")
    loss = discretized_mix_logistic_loss(y_hat,y)
    print(f"Loss : {loss}")
    