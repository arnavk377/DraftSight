"""Feature Tokenizer + Transformer (FT-Transformer) for tabular regression.

Each feature — numeric or categorical — is projected into a d_token-dimensional
token. A CLS token is prepended, standard transformer layers run over all tokens,
and the CLS output is regressed to a scalar.

Reference: Gorishniy et al. "Revisiting Deep Learning Models for Tabular Data"
           NeurIPS 2021.
"""

import torch
import torch.nn as nn


class _FeatureTokenizer(nn.Module):
    """Project each feature into a d_token-dimensional token.

    Numerics: element-wise linear per feature  (weight_i * x_i + bias_i).
    Categoricals: embedding lookup per feature.
    A learnable CLS token is prepended to the sequence.
    """

    def __init__(self, n_numeric: int, cat_cardinalities: list, d_token: int):
        super().__init__()
        self.num_weight = nn.Parameter(torch.empty(n_numeric, d_token))
        self.num_bias   = nn.Parameter(torch.zeros(n_numeric, d_token))
        self.cat_embeds = nn.ModuleList(
            [nn.Embedding(n, d_token) for n in cat_cardinalities]
        )
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.num_weight, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # x_num: (B, n_numeric) → (B, n_numeric, d_token)
        num_tok = x_num.unsqueeze(-1) * self.num_weight + self.num_bias
        # x_cat: (B, n_cat) → (B, n_cat, d_token)
        cat_tok = torch.stack(
            [self.cat_embeds[i](x_cat[:, i]) for i in range(x_cat.shape[1])], dim=1
        )
        tokens = torch.cat([num_tok, cat_tok], dim=1)       # (B, n_feat, d_token)
        cls = self.cls.expand(x_num.shape[0], -1, -1)       # (B, 1, d_token)
        return torch.cat([cls, tokens], dim=1)               # (B, 1+n_feat, d_token)


class _FTTLayer(nn.Module):
    def __init__(self, d_token: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn  = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        d_ffn = max(4 * d_token, 32)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_token),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class FTTransformer(nn.Module):
    """FT-Transformer: tokenize all features → transformer → regress from CLS.

    Args:
        n_numeric:        Number of (already-imputed) numeric input features.
        cat_cardinalities: Number of categories per categorical feature (including
                           an extra slot for unknowns).
        d_token:   Token dimensionality (must be divisible by n_heads).
        n_heads:   Number of attention heads.
        n_layers:  Number of transformer layers.
        dropout:   Dropout in attention and FFN.
    """

    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: list,
        d_token: int = 32,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_token % n_heads == 0, "d_token must be divisible by n_heads"
        self.tokenizer = _FeatureTokenizer(n_numeric, cat_cardinalities, d_token)
        self.layers = nn.ModuleList(
            [_FTTLayer(d_token, n_heads, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.ReLU(),
            nn.Linear(d_token, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x_num, x_cat)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x[:, 0])).squeeze(-1)  # regress from CLS token
