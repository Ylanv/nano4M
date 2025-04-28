import torch
from nanofm.utils.checkpoint import load_model_from_safetensors

# Constants from notebook
ckpt_path = './outputs/nano4M/multiclevr_d6-6w512/checkpoint-final.safetensors'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_generate_one_modality_roar_runs():
    # Load model
    model = load_model_from_safetensors(ckpt_path, device=device)
    model.eval()

    # Dummy inputs based on expected structure
    B = 1  # batch size
    L = 16  # token length
    x_tokens = torch.randint(0, model.vocab_size, (B, L)).to(device)
    x_positions = torch.arange(L).unsqueeze(0).expand(B, -1).to(device)
    x_modalities = torch.zeros(B, L, dtype=torch.long).to(device)  # all same modality

    # Generation parameters
    target_mod = 'scene_desc'  # can be 'tok_rgb@256', etc.
    num_steps, temp, top_p, top_k = 8, 0.7, 0.9, 0.0

    # Run generation
    pred_tokens, new_x_tokens, new_x_positions, new_x_modalities = model.generate_one_modality_roar(
        x_tokens, x_positions, x_modalities,
        target_mod=target_mod,
        num_steps=num_steps,
        temp=temp,
        top_p=top_p,
        top_k=top_k,
    )

    # Assertions
    assert isinstance(pred_tokens, torch.Tensor)
    assert pred_tokens.shape[0] == B
    assert pred_tokens.ndim == 2  # [B, num_pred_tokens]
