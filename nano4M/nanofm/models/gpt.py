# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from nanofm.modeling.transformer_layers import TransformerTrunk, LayerNorm
from nanofm.utils.sampling import sample_tokens


class GPT(nn.Module):
    """Basic autoregressive Transformer definition.

    Args:
        seq_read_key: Key in the data_dict corresponding to the full sequence
        dim: Transformer dimension
        depth: Number of transformer layers
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
        vocab_size: Size of the vocabulary
        max_seq_len: Maximum sequence length
        padding_idx: Index of the padding token, [PAD], in the vocabulary
        init_std: Standard deviation for weight initialization
    """

    def __init__(
        self,
        seq_read_key: str = "input_ids",
        dim: int = 512,
        depth: int = 8,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        use_bias: bool = False,
        vocab_size: int = 10000,
        max_seq_len: int = 256,
        padding_idx: int = 0,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.dim = dim
        self.seq_read_key = seq_read_key
        self.padding_idx = padding_idx
        self.max_seq_len = max_seq_len
        self.init_std = init_std
        # Embedding and positional layer
        self.input_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=dim, padding_idx=padding_idx
        )
        self.positional_embedding = nn.Parameter(torch.zeros(max_seq_len, dim))

        # Transformer
        self.trunk = TransformerTrunk(
            dim=dim,
            depth=depth,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            use_bias=use_bias,
        )

        # Normalization layer and projection to Vocab_size one hot encoding
        self.out_norm = LayerNorm(normalized_shape=dim, bias=use_bias)
        self.to_logits = nn.Linear(in_features=dim, out_features=vocab_size, bias=False)

        self.initialize_weights()  # Weight initialization

    @property
    def device(self):
        return next(self.parameters()).device

    def initialize_weights(self):
        """Initialize the weights of the model."""
        self.apply(self._init_weights)  # Initialize nn.Linear and nn.Embedding
        nn.init.normal_(
            self.positional_embedding, mean=0.0, std=self.init_std
        )  # Initialize the positional embeddings
        nn.init.constant_(self.to_logits.weight, 0)  # Zero-init the output projection

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.

            non_embedding: For non-embedding count (default), the input and output embeddings get subtracted.
        Args:
        Returns:
            The number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.input_embedding.weight.numel()
            n_params -= self.positional_embedding.numel()
            n_params -= self.to_logits.weight.numel()
        return n_params

    def forward_model(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: LongTensor of shape (B, L) containing the input token indices.
        Returns:
            Tensor of shape (B, L, vocab_size) containing the logits.
        """
        B, L = x.shape  # batch size and sequence length

        # Embed the input tokens using the input embedding layer. Shape: [B, L, D]
        embed = self.input_embedding(x)
        assert embed.shape == (
            B,
            L,
            self.dim,
        ), f"Expected embedding shape ({B},{L}, {self.dim}), but got {embed.shape}"

        # Add the positional embeddings to the tokens
        # Hint: Make sure this works for sequences of different lengths
        pos_embed = self.positional_embedding[:L, :]

        embed = embed + pos_embed

        # TODO: Define the causal mask for the transformer trunk.
        # False = masked-out, True = not masked. Shape: [1, L, L]
        # Hint: What shape should the mask have such that each token can attend to itself and
        # all previous tokens, but not to any future tokens?
        # Mask matrice should have upper part set as False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = (
            torch.ones(L, L, dtype=torch.bool).tril(diagonal=0).unsqueeze(0).to(device)
        )

        # Forward pass through Transformer trunk
        # Hint: Make sure to pass the causal mask to the transformer trunk too
        attn = self.trunk(embed, mask)

        # Pass to the output normalization and output projection layer to compute the logits
        norm = self.out_norm(attn)

        logit = self.to_logits(norm)
        return logit

    def compute_ce_loss(
        self,
        logits: torch.Tensor,
        target_seq: torch.LongTensor,
        padding_idx: int = -100,
    ) -> torch.Tensor:
        """
        Compute the cross-entropy loss given logits and target labels, ignoring padding tokens.

        Args:
             logits: Tensor of shape (B, L, vocab_size)
             target_seq: Tensor of shape (B, L) containing the target token indices.
             padding_idx: The index of the [PAD] token that should be ignored in the loss computation.
        Returns:
             A scalar loss value.
        """
        # TODO: Compute the cross-entropy loss
        # Hint: Remember to ignore the padding token index in the loss calculation

        # Reshape logits to shape (B * L, vocab_size) to fit the expected input for CrossEntropyLoss
        logits = logits.reshape(-1, logits.size(-1))  # Flatten the logits tensor
        target_seq = target_seq.reshape(-1)  # Flatten the target sequence tensor

        # Initialize the loss function
        criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)

        # Compute the cross-entropy loss, ignoring the padding token
        loss = criterion(logits, target_seq)
        return loss

    def forward(self, data_dict: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the model.

        Args:
            data_dict: A dictionary containing the input sequence.
        Returns:
            The loss and a dictionary containing the perplexity metric.
        """
        seq = data_dict[
            self.seq_read_key
        ]  # Shape (B, L+1): e.g. [SOS], T_1, T_2, ..., T_L, [EOS], [PAD] ...
        input_seq = seq[
            :, :-1
        ]  # Shape (B, L): e.g. [SOS], T_1, T_2, ..., T_L, [EOS], [PAD] ... (with the last token dropped)
        target_seq = seq[
            :, 1:
        ]  # Shape (B, L): e.g. T_1, T_2, ..., T_L, [EOS], [PAD] ... (with the first token dropped)

        # Forward pass through the model and compute loss
        logits = self.forward_model(input_seq)
        loss = self.compute_ce_loss(logits, target_seq, padding_idx=self.padding_idx)

        metrics_dict = {"ppl": torch.exp(loss)}  # Perplexity
        return loss, metrics_dict

    @torch.no_grad()
    def generate(
        self,
        context: List[int] = [0],
        eos_idx: Optional[int] = None,
        temp: float = 1.0,
        top_p: float = 0.0,
        top_k: float = 0.0,
    ) -> torch.Tensor:
        """
        Generate a sequence autoregressively given an initial context.

        Args:
            context: List of token indices to start the sequence.
            eos_idx: Optional [EOS] index of the end-of-sequence token. If None, the
                model will generate until the maximum sequence length.
            temp: Temperature for sampling.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling threshold.
        Returns:
            A tensor of shape (1, L) containing the generated sequence.
        """
        was_training = self.training
        self.eval()

        # Initialize the sequence with the start-of-sequence token
        current_tokens = torch.tensor([context], dtype=torch.long, device=self.device)
        for _ in range(self.max_seq_len - len(context)):

            # Run a forward pass through the model to get the logits
            logits = self.forward_model(current_tokens)  # Shape (B,L,vocab_size)

            # Keep only the last token's logits and sample the next token
            logits = logits[:, -1, :]
            # Hint: Use the sample_tokens function from utils/sampling.py
            # Make sure to pass the temperature, top_k and top_p arguments
            samples, sampled_probs = sample_tokens(
                logits, temperature=temp, top_k=top_k, top_p=top_p
            )

            # Concatenate the new token to the current_tokens sequence
            current_tokens = torch.cat((current_tokens, samples.unsqueeze(1)), dim=1)

            # Break if the end-of-sequence token is generated
            if eos_idx == samples:
                break

        if was_training:
            self.train()

        return current_tokens
