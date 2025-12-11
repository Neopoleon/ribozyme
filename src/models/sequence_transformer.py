"""Sequence-only Transformer baseline for RNA classification."""

from __future__ import annotations

import torch
from torch import nn


class RNASequenceTransformer(nn.Module):
    """Standard Transformer encoder stacked on token + positional embeddings."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        pad_token_id: int = 5,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T] padded token ids
            attention_mask: [B, T] bool mask, True for valid tokens
        """
        if input_ids.size(1) > self.max_seq_len:
            raise ValueError(
                f"Sequence length {input_ids.size(1)} exceeds max_seq_len={self.max_seq_len}."
            )

        positions = torch.arange(
            0, input_ids.size(1),
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.encoder(
            x,
            src_key_padding_mask=~attention_mask,
        )

        # Masked mean pooling over time dimension
        mask = attention_mask.unsqueeze(-1).type_as(x)
        masked_sum = (x * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        pooled = masked_sum / lengths

        pooled = self.layer_norm(self.dropout(pooled))
        logits = self.classifier(pooled)
        return logits
