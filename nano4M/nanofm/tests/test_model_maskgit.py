import torch
import pytest
from nanofm.models.gpt import GPT

import re
from einops import rearrange
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import LinearSegmentedColormap
import math
from nanofm.utils.checkpoint import load_model_from_safetensors
from nanofm.data.vision.tokenized_mnist import (
    create_tokenized_mnist_dataloader,
    detokenize_MNIST,
)
from nanofm.models.maskgit import MaskGIT
import os

# Define the device (use GPU if available, otherwise fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice : {device}\n")

# Parameters for the tests
vocab_size = 10000
dim = 512
max_seq_len = 20
padding_idx = 0
eos_token_id = 4  # Assume 4 is the EOS token ID
batch_size = 2
sequence_length = 5


@pytest.fixture
def model():
    """Fixture to initialize and return the model for each test"""
    return MaskGIT(seq_len=sequence_length, vocab_size=vocab_size, dim=dim).to(device)


@pytest.fixture
def mask(model):
    mask = model.generate_random_mask(torch.rand(batch_size, sequence_length))
    return mask


@pytest.fixture
def model_train():
    ckpt_path = "./outputs/nanoMaskGIT/mnist_d8w512/checkpoint-final.safetensors"
    model = load_model_from_safetensors(ckpt_path, device=device)
    print(f"{model.get_num_params() / 10**6}M parameters")
    return model


def test_forward_model(model, mask):
    x = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)
    logits = model.forward_model(x, mask)
    assert logits.shape == (
        batch_size,
        sequence_length,
        vocab_size,
    ), "Failed test_forward_model"


def test_get_maskgit_schedule(model, mask):

    # Simulate a mask with 17 masked tokens
    L = sequence_length
    num_masked = 2
    num_steps = sequence_length

    schedule = model.get_maskgit_schedule(mask, num_steps=num_steps)

    assert isinstance(schedule, list), "Schedule should be a list"
    assert len(schedule) == num_steps, "Schedule length should match num_steps"
    print(schedule)


def test_generate(model):
    seq = torch.randint(0, vocab_size, (sequence_length,), device=device)
    mask = torch.tensor(
        [
            True,
            False,
            False,
            True,
            False,
        ]
    ).to(device=device)
    gen_seq = model.generate(seq, mask, num_steps=2)
    print(gen_seq)


def test_generate_history(model_train):
    seq = torch.zeros(50, dtype=torch.long, device=device)
    mask = torch.ones(50, dtype=torch.bool, device=device)
    seq[0] = 9
    mask[0] = False

    _, mask_h = model_train.generate(seq, mask, num_steps=2, return_history=True)
    print(mask_h.shape)
    print(mask_h)
    pattern = get_unmasking_steps(mask_h)
    print(pattern.shape)
    print(pattern)
    visualize_tensor_as_color_grid(pattern.reshape(10, 5))
    # plot_unmasking_pattern(pattern,"nanofm/tests/visu/pattern_img_txt.png")


def test_abla_nstep(model_train):
    generate_samples(model_train, num_steps=8, temp=0.7, top_p=0.9, top_k=0.0)


def generate_samples(model, num_steps=8, temp=1.0, top_p=0.0, top_k=0.0, n_samples=10):
    k_l = [1, 4, 8, 16, 32, 49]
    fig = plt.figure(figsize=(8.0, 8.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(len(k_l), n_samples), axes_pad=(0.1, 0.7))
    for idx, k in enumerate(k_l):
        for sample_idx in range(n_samples):
            grid_idx = idx * n_samples + sample_idx

            seq = torch.zeros(50, dtype=torch.long, device=device)
            mask = torch.ones(50, dtype=torch.bool, device=device)
            seq[0] = 0
            mask[0] = False
            output = model.generate(
                seq,
                mask,
                num_steps=k,
                temp=temp,
                top_p=top_p,
                top_k=top_k,
                return_history=False,
            )

            reconst = detokenize_MNIST(
                output, patch_size=2, account_for_labels=True
            ).cpu()
            grid[grid_idx].imshow(reconst[0], cmap="Greys", vmin=0, vmax=1)
            if sample_idx == 0:
                grid[grid_idx].set_title(f"k = {k}", fontsize=10)

    save_path = "debug_output.png"
    plt.savefig(save_path)
    print(f"Saved debug image to: {save_path}")
    plt.close(fig)


@pytest.fixture
def model_train_ts():
    ckpt_path = "./outputs/nanoMaskGIT/tinystories_d8w512/checkpoint-final.safetensors"
    model = load_model_from_safetensors(ckpt_path, device=device)
    print(f"{model.get_num_params() / 10**6}M parameters")
    return model


@pytest.fixture
def tokenizer():
    # Load the GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)

    # Add padding, start-of-sequence, and end-of-sequence tokens
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.add_special_tokens(
        {
            "bos_token": "[SOS]",
            "eos_token": "[EOS]",
        }
    )
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="[SOS] $A [EOS]",
        special_tokens=[
            ("[EOS]", tokenizer.eos_token_id),
            ("[SOS]", tokenizer.bos_token_id),
        ],
    )
    return tokenizer


def token_ids_to_text(token_ids, text_tokenizer):
    """Helper function to turn token sequences back to well-formatted text."""
    decoded = text_tokenizer.decode(token_ids)
    # Remove [SOS], [EOS], and [PAD] tokens along with surrounding horizontal whitespace only.
    decoded = re.sub(r"[ \t]*\[(SOS|EOS|PAD)\][ \t]*", " ", decoded)
    # Collapse extra horizontal spaces in each line without touching newline characters.
    decoded = "\n".join(
        [re.sub(r"[ \t]+", " ", line).strip() for line in decoded.splitlines()]
    )
    return decoded


def test_generate_txt(model_train_ts, tokenizer):
    for _ in range(5):
        seq = torch.zeros(256, dtype=torch.long, device=device)
        mask = torch.ones(256, dtype=torch.bool, device=device)
        output = model_train_ts.generate(
            seq, mask, num_steps=128, temp=0.5, top_k=100, return_history=False
        )
        print(output.shape)
        print(output)
        print(token_ids_to_text(output[0], text_tokenizer=tokenizer))
        print("\n" + "-" * 50 + "\n")


def test_generate_txt_history(model_train_ts, tokenizer):
    seq = torch.zeros(256, dtype=torch.long, device=device)
    mask = torch.ones(256, dtype=torch.bool, device=device)
    seq_h, mask_h = model_train_ts.generate(
        seq, mask, num_steps=128, temp=1.0, top_k=100, return_history=True
    )
    print(
        seq_h.shape
    )  # should be #Step * seq_len (256) where a value is between 0 and vocab_size i.e 128 x 256
    print(seq_h)
    print(mask_h)

    for idx, seq_id in enumerate(seq_h):
        if idx > 10:
            break
        print("\n" + "-" * 50 + "\n")
        print(token_ids_to_text(seq_id, text_tokenizer=tokenizer))
        print(mask_h[idx])


def test_generate_txt_history_pattern(model_train_ts, tokenizer):
    seq = torch.zeros(256, dtype=torch.long, device=device)
    mask = torch.ones(256, dtype=torch.bool, device=device)
    _, mask_h = model_train_ts.generate(
        seq, mask, num_steps=128, temp=1.0, top_k=100, return_history=True
    )
    prediction_pattern = get_unmasking_steps(mask_h)
    plot_unmasking_pattern(prediction_pattern)
    print(prediction_pattern)


def get_unmasking_steps(mask_history: torch.BoolTensor) -> torch.LongTensor:
    """
    Returns a tensor of shape (seq_length,) where each value represents the step
    at which that token was first unmasked (i.e., went from True to False).

    Args:
        mask_history: Tensor of shape (num_steps, seq_length), with boolean values.
                      True = still masked, False = unmasked.

    Returns:
        Tensor of shape (seq_length,), with step indices where unmasking occurred.
    """
    num_steps, seq_len = mask_history.shape
    step_of_prediction = torch.full((seq_len,), fill_value=-1, dtype=torch.long)

    for step in range(1, num_steps):  # Start from step 1 (step 0 usually all True)
        prev_mask = mask_history[step - 1]
        curr_mask = mask_history[step]
        # Tokens that went from True -> False were unmasked at this step
        newly_unmasked = prev_mask & ~curr_mask
        step_of_prediction[newly_unmasked] = step

    return step_of_prediction


def plot_unmasking_pattern(
    step_of_prediction: torch.Tensor,
    save_path: str = "nanofm/tests/visu/pattern_txt.png",
):
    """
    Saves a clean visualization of token prediction steps as a horizontal color strip.

    Args:
        step_of_prediction: Tensor of shape (seq_length,) where each value is the step
                            at which a token was unmasked.
        save_path: Where to save the output image.
    """
    seq_length = step_of_prediction.size(0)
    steps = step_of_prediction.cpu().numpy()

    # Custom colormap: green -> yellow -> red
    cmap = LinearSegmentedColormap.from_list("green_to_red", ["green", "yellow", "red"])

    fig, ax = plt.subplots(figsize=(12, 1.5))

    # Display color-coded strip
    im = ax.imshow(steps[np.newaxis, :], cmap=cmap, aspect="auto")

    # Remove all ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Unmasking steps (Green = early, Red = late)", fontsize=12)

    # Add colorbar as legend
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal")
    cbar.set_label("Prediction Step")

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    print(f"✅ Saved clean unmasking pattern to: {save_path}")
    plt.close(fig)


def visualize_tensor_as_color_grid(
    tensor_2d: torch.Tensor, save_path="nanofm/tests/visu/color_grid.png"
):
    """
    Visualizes a 2D tensor as a color-coded grid where each value is mapped to a color.

    Args:
        tensor_2d (torch.Tensor): A 2D tensor of shape (H, W)
        save_path (str): Path to save the visualization
    """

    # Convert to numpy
    grid = tensor_2d.cpu().numpy()

    # Define a smooth green → yellow → red gradient
    cmap = LinearSegmentedColormap.from_list("green_to_red", ["green", "yellow", "red"])

    fig, ax = plt.subplots(figsize=(3, 6))  # adjust size as needed
    im = ax.imshow(grid, cmap=cmap, vmin=grid.min(), vmax=grid.max())

    # Remove ticks and labels for a clean look
    ax.set_xticks([])
    ax.set_yticks([])

    # Add colorbar to show value → color mapping
    plt.colorbar(im, ax=ax, orientation="vertical", label="Value")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    print(f"✅ Saved color grid to: {save_path}")
    plt.close(fig)
