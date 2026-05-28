"""MLP with entity embeddings for tabular regression.

Categorical features are passed through learned dense embeddings instead of
one-hot encoding, which gives the model a compact, trainable representation
of high-cardinality columns (position, college, team, etc.).
"""

import torch
import torch.nn as nn


def _embed_dim(n_cats: int) -> int:
    """Standard heuristic: min(50, max(2, (n_cats + 1) // 2))."""
    return min(50, max(2, (n_cats + 1) // 2))


class MLPWithEmbeddings(nn.Module):
    """MLP that concatenates entity embeddings with numeric features.

    Args:
        n_numeric:        Number of (already-imputed) numeric input features.
        cat_cardinalities: Number of categories per categorical feature (including
                           an extra slot for unknowns).
        hidden_dims:      Sizes of hidden layers.
        dropout:          Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: list,
        hidden_dims: tuple = (256, 128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        embed_dims = [_embed_dim(n) for n in cat_cardinalities]
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, d) for n, d in zip(cat_cardinalities, embed_dims)]
        )
        in_dim = n_numeric + sum(embed_dims)
        layers = []
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        embeds = [self.embeddings[i](x_cat[:, i]) for i in range(x_cat.shape[1])]
        x = torch.cat([x_num] + embeds, dim=-1)
        return self.mlp(x).squeeze(-1)
