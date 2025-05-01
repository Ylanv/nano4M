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

from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from nanofm.modeling.transformer_layers import TransformerTrunk, LayerNorm
from nanofm.utils.sampling import sample_tokens


class MaskGIT(nn.Module):
    """
    MaskGIT model implementation using a full bi-directional Transformer.

    Given a full input sequence, the model randomly masks out a number of tokens
    (between 1 and L, per sample) by replacing them with a learned mask token.
    The loss is computed only on the non-masked tokens using cross-entropy.

    Args:
        seq_read_key: Key in the input dictionary for the full sequence (token IDs).
        dim: Transformer dimension.
        depth: Number of transformer layers.
        head_dim: Dimension of each attention head.
        mlp_ratio: Ratio of the MLP hidden dimension to the transformer dimension.
        use_bias: Whether to include bias in QKV, attention projections and MLP layers.
        vocab_size: Vocabulary size (should include extra tokens for class conditioning if needed).
        seq_len: Sequence length expected (for learned positional embeddings).
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
        seq_len: int = 256,
        init_std: float = 0.02,
    ):
        super().__init__()

        self.dim = dim
        self.vocab_size = vocab_size
        self.seq_read_key = seq_read_key
        self.init_std = init_std

        self.input_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=dim
        )
        self.positional_embedding = nn.Parameter(torch.zeros(seq_len, dim))
        self.mask_token = nn.Parameter(torch.zeros(dim))

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
        nn.init.normal_(
            self.mask_token, mean=0.0, std=self.init_std
        )  # Initialize the mask token
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

        Args:
            non_embedding: For non-embedding count (default), the input and output embeddings get subtracted.
        Returns:
            The number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.input_embedding.weight.numel()
            n_params -= self.positional_embedding.numel()
            n_params -= self.mask_token.numel()
            n_params -= self.to_logits.weight.numel()
        return n_params

    def forward_model(
        self, x: torch.LongTensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Embeds the input tokens, replaces masked positions with the learned mask token,
        adds positional embeddings, and passes through the transformer trunk.

        Args:
            x: Tensor of shape (B, L) with token IDs.
            mask: Boolean tensor of shape (B, L) where True indicates a masked token.
        Returns:
            Logits tensor of shape (B, L, vocab_size).
        """

        B, L = x.size()  # batch size and sequence length
        assert mask.shape == (B, L), f"Mask shape should be ({B},{L}) got {mask.shape}"

        # Embed the input tokens using the input embedding layer. Shape: [B, L, D]
        embed = self.input_embedding(x)
        assert embed.shape == (
            B,
            L,
            self.dim,
        ), f"Expected embedding shape ({B},{L}, {self.dim}), but got {embed.shape}"

        # Replace embeddings for masked tokens with the learned self.mask_token, wherever mask is True.
        # The mask token (D) is broadcast to all masked positions (B, L)
        embed[mask] = self.mask_token

        # Add the positional embeddings to the tokens
        embed = embed + self.positional_embedding[:L, :]

        # Forward pass through Transformer trunk
        # Hint: No causal mask is needed here, since we are using full self-attention.
        att = self.trunk(embed)

        # Pass to the output normalization and output projection layer to compute the logits
        norm = self.out_norm(att)

        logits = self.to_logits(norm)
        assert logits.shape == (
            B,
            L,
            self.vocab_size,
        ), f"Expected logits shape ({B},{L}, {self.vocab_size}), but got {logits.shape}"
        # Return the logits
        return logits

    def generate_random_mask(self, seq: torch.Tensor) -> torch.BoolTensor:
        """
        Generates a random mask for each sample in the batch.
        Each sample has a random number of tokens (between 1 and L)
        that are masked (True) and the rest are not masked (False).

        Args:
            seq: Tensor of shape (B, L) with token IDs.
        Returns:
            A boolean tensor of shape (B, L) where True indicates a masked token.
        """
        B, L = seq.size()

        # TODO: Generate and return a random mask of shape (B, L), where
        # True = masked-out, False = not masked. Each sample should have a
        # random number of masked-out tokens between 1 and L. The mask shouldgenerate_random_mask(
        # be generated such that the number of masked tokens is different
        # for each sample in the batch.
        # Note: How can you avoid using a for loop here, and instead use
        # vectorized operations?
        # Hint: Don't forget to create the mask on the same device as seq.

        device = seq.device
        # Generate a number in [1,L] for each sequence in the batch. (i.e B times) and unsqueeze it => [[R1],[R2],...,[RB]] where R is a random bumber
        num_masked = torch.randint(1, L + 1, (B,), device=device).unsqueeze(1)

        # Generate random tensor of size (B,L)
        random_tensor = torch.rand(B, L, device=device)

        # For each sample we take the n-smallest point, where n is randomly generated for each sample in the batch and set them to True
        _, indices = random_tensor.sort()

        # Now we have a tensor of indices (B,L) sorted by order => for the first num_masked[i] we set to True.
        # Create a range of size L and expand it to B => [[1...L],[1...L],...,[1...L]]
        range = torch.arange(L, device=device).expand(B, L)
        # Create boolean mask using the range and the num_masked
        mask = range < num_masked

        # Now we need to "undo" the sort
        final_mask = torch.gather(mask, dim=1, index=indices)
        assert final_mask.shape == (
            B,
            L,
        ), f"Mask should be of shape : ({B},{L}) , got {final_mask.shape}"

        return final_mask

    def compute_ce_loss(
        self,
        logits: torch.Tensor,
        target_seq: torch.LongTensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        Compute the cross-entropy loss given logits and target labels, ignoring masked target tokens.

        Args:
             logits: Tensor of shape (B, L, vocab_size)
             target_seq: Tensor of shape (B, L) containing the target token indices.
             ignore_index: The token index that should be ignored in the loss computation.
        Returns:
             A scalar loss value.
        """

        # Hint: Remember to ignore the ignore_index in the loss calculation

        # Reshape logits to shape (B * L, vocab_size) to fit the expected input for CrossEntropyLoss
        logits = logits.reshape(-1, logits.size(-1))  # Flatten the logits tensor
        target_seq = target_seq.reshape(-1)  # Flatten the target sequence tensor

        # Initialize the loss function
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

        # Compute the cross-entropy loss, ignoring the padding token
        loss = criterion(logits, target_seq)
        return loss

    def forward(self, data_dict: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass for training.
        Randomly selects a number of tokens (between 1 and L) to mask per sample,
        replaces them with the learned mask token, and computes the cross-entropy loss
        only on the non-masked tokens.

        Args:
            data_dict: Dictionary containing the input sequence.
        Returns:
            The loss and a dictionary containing the perplexity metric.
        """
        # Get the full input sequence, shape (B, L)
        seq = data_dict[self.seq_read_key]

        # Generate a random mask for each sample. True = masked-out, False = not masked
        mask = self.generate_random_mask(seq)

        # Prepare targets: for masked-out positions, target is the original token.
        # For non-masked positions, set to some ignore index (-100) so loss is not computed for input tokens.
        target = seq.clone()
        target[~mask] = -100

        # Forward pass through the model and compute loss
        logits = self.forward_model(seq, mask)
        loss = self.compute_ce_loss(logits, target, ignore_index=-100)

        metrics_dict = {"ppl": torch.exp(loss)}  # Perplexity
        return loss, metrics_dict

    def get_maskgit_schedule(
        self, mask: torch.BoolTensor, num_steps: int = 8
    ) -> List[int]:
        """
        Generates a MaskGIT schedule for unmasking tokens at inference time. We only added a
        constant schedule for now, but feel free to add more schedules, e.g. a cosine schedule!

        Args:
            mask: Boolean tensor of shape (L,) where True indicates a masked-out token.
            num_steps: Number of steps to unmask tokens.
        Returns:
            A list of integers representing the number of tokens to unmask at each step.
        """
        # Get total number of tokens to unmask
        total_tokens = int(mask.sum().item())

        assert total_tokens > 0, "No tokens to unmask in the input sequence."
        assert num_steps > 0, "Number of steps should be greater than zero."
        assert (
            num_steps <= total_tokens
        ), "Number of steps should be less than or equal to the total number of tokens to unmask."

        # Implement a constant schedule, where you unmask a constant number of
        # tokens at each step. The mask of shape (L,) defines the number of tokens to unmask.
        # For example, if total_tokens = 17 and num_steps = 8, then the schedule should be:
        # [2, 2, 2, 2, 2, 2, 2, 3]. If the total number of tokens is not divisible by the
        # number of steps, we simply add the remainder to the last step.
        # The `schedule` should be a list of integers of length `num_steps`, where each integer
        # represents the number of tokens to unmask at that step. The sum of the integers in
        # `schedule` should equal `total_tokens`.
        schedule = [total_tokens // num_steps for _ in range(num_steps)]
        schedule[-1] += total_tokens % num_steps

        assert (
            len(schedule) == num_steps
        ), "Schedule length should match the number of steps."
        assert (
            sum(schedule) == total_tokens
        ), "Total number of tokens to unmask should match the sum of the schedule."

        return schedule

    @torch.no_grad()
    def generate(
        self,
        seq: torch.LongTensor,
        mask: torch.BoolTensor,
        num_steps: int = 8,
        temp: float = 1.0,
        top_p: float = 0.0,
        top_k: float = 0.0,
        return_history: bool = False,
    ) -> torch.Tensor:
        """
        Generate a sequence through iterative unmasking, using the MaskGIT schedule.

        Args:
            seq: Tensor of shape (L,) with token IDs. Wherever mask is True, the corresponding
                entries in seq can be any value, e.g. random. They will be replaced during
                the generation process.
            mask: Boolean tensor of shape (L,) where True indicates a masked-out token.
            num_steps: Number of MaskGIT decoding steps.
            temp: Temperature for sampling.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling threshold.
            return_history: Whether to return the history of generated sequences and masks.
        Returns:
            A tensor of shape (L,) containing the generated sequence.
            If return_history is True, returns a tuple of (seq_history, mask_history).
        """
        was_training = self.training
        self.eval()

        L = seq.size(0)
        assert mask.dim() == 1 and mask.size(0) == L

        # Get schedule for unmasking tokens
        schedule = self.get_maskgit_schedule(mask, num_steps)

        # Add batch dimension to sequence and mask
        seq = seq.unsqueeze(0)  # shape (1, L)
        mask = mask.unsqueeze(0)  # shape (1, L)

        if return_history:
            seq_history, mask_history = [seq.clone().cpu()], [mask.clone().cpu()]

        for step, k in enumerate(schedule):
            # Forward pass through the model to get the logits. Shape: [1, L, vocab_size]
            logits = self.forward_model(seq, mask)
            assert logits.shape == (
                1,
                L,
                self.vocab_size,
            ), f"Logits shape should be (1,{L},{self.vocab_size}), got {logits.shape}"

            # Get the indices of masked tokens. Shape: [M,] (M = number of masked tokens) Note : 0 = False, else = 1
            masked_indices = torch.nonzero(mask[0], as_tuple=False).squeeze(1)

            # Get the logits for the `masked_indices` positions. Shape: [M, vocab_size]
            masked_logits = logits[
                0, masked_indices
            ]  # Note : we can do tensor[tensor] to select index from a list (like in numpy)

            assert (
                masked_logits.dim() == 2
            ), f"Masked logits should have {2} dim, got {masked_logits.dim()}"

            # Compute confidence scores from `masked_logits`. Shape: [M,]
            # Hint: As a proxy for confidence, we use the maximum logit value for each masked position.
            confidence = torch.max(masked_logits, dim=-1).values

            # Based on the number of tokens `k` to unmask at this step in the schedule,
            # select the top-k masked positions based on confidence. Shape: [k,]
            # Hint: First, get the top-k indices of the confidence scores, and then use these indices
            # to select the corresponding masked positions.

            # Sort the confidence list
            _, sorted_index = torch.sort(confidence, descending=True)

            topk_index = sorted_index[:k]
            # Take top-k
            selected_positions = masked_indices[topk_index]

            # Get the logits for the `selected_positions`. Shape: [k, vocab_size]
            selected_logits = logits[0, selected_positions]
            assert selected_logits.shape == (
                k,
                self.vocab_size,
            ), f"Selected logits shape should be ({k},{self.vocab_size}), got {selected_logits.shape}"

            # Sample new tokens for the selected_positions
            # Hint: Use the sample_tokens function from utils/sampling.py
            # Make sure to pass the `temp`, `top_k` and `top_p` arguments
            samples, _ = sample_tokens(
                selected_logits, temperature=temp, top_k=top_k, top_p=top_p
            )

            # TODO: Update the sequence and mask.
            # Replace the selected positions in `seq` with the sampled tokens
            # and set the corresponding positions in `mask` to False (indicating that
            # these positions are no longer masked).
            seq[0, selected_positions] = samples
            mask[0, selected_positions] = False

            if return_history:
                seq_history.append(seq.clone().cpu())
                mask_history.append(mask.clone().cpu())

        if was_training:
            self.train()

        if return_history:
            # Concatenate the history of sequences and masks and return them
            return torch.cat(seq_history, dim=0), torch.cat(mask_history, dim=0)

        # Return the generated sequence
        return seq
