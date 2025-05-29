import os
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

DATA_PATH = "/work/com-304/snoupy/audiocaps/train/audio/13003.wav"
TARGET_SR = 48000  # EnCodec 48kHz model

def load_tokenizer():
    model = EncodecModel.encodec_model_48khz().to("cuda")
    model.set_target_bandwidth(6.0)
    model.eval()
    return model.to("cuda")

def tokenize_audio(model, wav_path):
    # Load audio
    wav, sr = torchaudio.load(wav_path)  # wav: [C, T]
    wav = wav.to("cuda")

    # Convert to mono if needed
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Convert to target sample rate and 1 channel
    wav = convert_audio(wav, sr,model.sample_rate, model.channels)

    # Encode and decode
    with torch.no_grad():
        encoded_frames = model.encode(wav)  # List[Frame]
        decoded_audio = model.decode(encoded_frames)  # [1, 1, T]
    
    # Extract tokens (optional: flatten across codebooks)
    codes = torch.cat([frame.codes for frame in encoded_frames], dim=-1)  # [1, N_codebooks, T_codes]
    return codes.squeeze(0).cpu().numpy(), decoded_audio.squeeze(0).cpu()

def main():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Instantiate a pretrained EnCodec model
    model = EncodecModel.encodec_model_48khz()
    # The number of codebooks used will be determined bythe bandwidth selected.
    # E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
    # Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
    # For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
    # of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
    model.set_target_bandwidth(6.0)

    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(DATA_PATH)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = model.encode(wav)
        decoded = model.decode(encoded_frames)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
    print(f"Decoded shape : {decoded.shape}")  
    print(f"Codes shape : {codes.shape}")
    pa = "/home/vifian/nano4M/audio_tokenizer/encodec/"
    torchaudio.save(f"{pa}/original.wav",wav.squeeze(0),model.sample_rate)
    torchaudio.save(f"{pa}/reconstructed.wav",decoded.squeeze(0),model.sample_rate)
if __name__ == "__main__":
    main()
