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

from nanofm.utils.checkpoint import load_model_from_safetensors
from nanofm.data.vision.tokenized_mnist import (
    create_tokenized_mnist_dataloader,
    detokenize_MNIST,
)


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
sequence_length = 10

# Sample input
context = [1, 2, 3]  # Example context (start-of-sequence tokens)


@pytest.fixture
def model():
    """Fixture to initialize and return the model for each test"""
    return GPT(
        vocab_size=vocab_size, max_seq_len=max_seq_len, padding_idx=padding_idx
    ).to(device)


def test_forward(model):
    """Test the forward pass"""
    # Create a random input tensor of shape [B, L] (batch_size, sequence_length)
    x = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)

    # Perform the forward pass
    logits = model.forward_model(x)

    # Check if the output shape is correct [B, L, vocab_size]
    assert logits.shape == (
        batch_size,
        sequence_length,
        vocab_size,
    ), f"Expected shape [B, L, vocab_size], but got {logits.shape}"


def test_compute_ce_loss(model):
    """Test the compute_ce_loss function"""
    # Create random logits and target sequence
    logits = torch.rand(batch_size, sequence_length, vocab_size, device=device)
    target_seq = torch.randint(
        0, vocab_size, (batch_size, sequence_length), device=device
    )

    # Assume padding tokens are at index `padding_idx`
    target_seq[0, 3] = padding_idx  # Simulate padding token at position (0, 3)
    target_seq[1, 5] = padding_idx  # Simulate padding token at position (1, 5)

    # Compute the cross-entropy loss
    loss = model.compute_ce_loss(logits, target_seq, padding_idx)

    # Check if the loss is a scalar
    assert loss.item() >= 0, f"Loss should be non-negative, but got {loss.item()}"


def test_generate_sequence(model):
    """Test the sequence generation with EOS token"""
    # Create the model with EOS token
    eos_token_id = 4

    # Generate sequence starting from the context
    generated_sequence = model.generate(context, eos_token_id)

    # Check if the generated sequence contains the EOS token and stops
    assert (
        eos_token_id not in generated_sequence[:, -1].cpu().numpy()
    ), f"Expected EOS token to stop generation, but EOS token was found in the output."

    assert (
        generated_sequence.dim() == 2
    ), f"Expected sequence of dim 2, got {generated_sequence.dim()} instead"

    assert not torch.any(
        generated_sequence > vocab_size
    ), f"One of the token ID generated is out of the vocabulary"

    assert (
        generated_sequence.shape[1] <= max_seq_len
    ), f"The generated sequence is longer than max_seq_len"
    print(generated_sequence)


@pytest.fixture
def load_train_model():
    ckpt_path = "./outputs/nanoGPT/tinystories_d8w512/checkpoint-final.safetensors"
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


def test_generate_train(load_train_model, tokenizer):
    # Generate 5 story
    for _ in range(5):
        output = load_train_model.generate(
            context=[tokenizer.bos_token_id],
            temp=1.0,
            top_p=0.0,
            top_k=0.0,
            eos_idx=tokenizer.eos_token_id,
        )[0]
        print(token_ids_to_text(output, text_tokenizer=tokenizer))
        print("\n" + "-" * 50 + "\n")


def test_conditional_generate(load_train_model, tokenizer):
    context = tokenizer.encode("Daisy was hungry, so she")[
        :-1
    ]  # Encode and discard automatically added [EOS] token

    for _ in range(1):
        output = load_train_model.generate(
            context=context,
            temp=1.0,
            top_p=0.0,
            top_k=0.0,
            eos_idx=tokenizer.eos_token_id,
        )[0]
        print(token_ids_to_text(output, text_tokenizer=tokenizer))
        print("\n" + "-" * 50 + "\n")


def test_generate_temp(load_train_model, tokenizer):
    temp = np.arange(0.0, 1, 0.5)
    print("\n" + "-" * 50 + "\n")
    print(f"Temperature ablation: {temp}\n")
    print("\n" + "-" * 50 + "\n")
    for t in temp:
        print(f"Temp : {t}\n")
        output = load_train_model.generate(
            context=[tokenizer.bos_token_id],
            temp=t,
            top_p=0.0,
            top_k=0.0,
            eos_idx=tokenizer.eos_token_id,
        )[0]
        print(token_ids_to_text(output, text_tokenizer=tokenizer))
        print("\n" + "-" * 50 + "\n")


def test_MNIST():
    data_loader = create_tokenized_mnist_dataloader(
        train=False, add_label_token=True, shuffle=False
    )
    data_dict = next(iter(data_loader))

    tokens = data_dict["input_ids"]
    reconst = detokenize_MNIST(tokens, patch_size=2, account_for_labels=True)
    bits = rearrange(reconst, "b (nh ph) (nw pw) -> b (nh nw) ph pw", ph=2, pw=2)

    for i in range(2):
        plt.figure(figsize=(5.0, 5.0))
        plt.imshow(reconst[i], cmap="Greys")
        plt.show()

        fig = plt.figure(figsize=(5.0, 5.0))
        grid = ImageGrid(fig, 111, nrows_ncols=(7, 7), axes_pad=0.1)
        for j, img in enumerate(bits[i]):
            grid[j].imshow(img, cmap="Greys", vmin=0, vmax=1)
        plt.show()

        print("Tokens:", tokens[i], "\n\n")


@pytest.fixture
def load_mnist():
    ckpt_path = "./outputs/nanoGPT/mnist_d8w512/checkpoint-final.safetensors"
    model = load_model_from_safetensors(ckpt_path, device=device)
    print(f"{model.get_num_params() / 10**6}M parameters")
    return model


def test_generate_label(load_mnist, monkeypatch):
    label = 5
    output = load_mnist.generate(context=[label], temp=5, top_p=0.0)
    reconst = detokenize_MNIST(output, patch_size=2, account_for_labels=True).cpu()
    fig, ax = plt.subplots()
    ax.imshow(reconst[0], cmap="gray_r")
    save_path = "debug_output.png"
    plt.savefig(save_path)
    print(f"Saved debug image to: {save_path}")
    plt.close(fig)


def test_generate_all_label(load_mnist):
    top_k = 0.0
    n_samples = 10
    temperatures = 0.5
    top_p = np.linspace(0, 1, 10)  # 10 values from 0 to 5, one per label

    fig = plt.figure(figsize=(8.0, 8.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(10, n_samples), axes_pad=10)

    for label in range(10):
        top_pp = top_p[label]  # use a different temperature for each label
        for sample_idx in range(n_samples):
            grid_idx = label * n_samples + sample_idx
            output = load_mnist.generate(
                context=[label], temp=temperatures, top_p=top_pp, top_k=top_k
            )
            reconst = detokenize_MNIST(
                output, patch_size=2, account_for_labels=True
            ).cpu()
            grid[grid_idx].imshow(reconst[0], cmap="Greys", vmin=0, vmax=1)

    save_path = "debug_output.png"
    plt.savefig(save_path)
    print(f"Saved debug image to: {save_path}")
    plt.close(fig)
