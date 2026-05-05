"""
TabNet: Attentive Interpretable Tabular Learning
Ö. Arik & T. Pfister, AAAI 2021 (arXiv 1908.07442)

Architecture:
  - Ghost Batch Normalization on all internal layers
  - Feature transformer: N_shared shared FC-BN-GLU layers + N_step_dep step-specific layers
  - Attentive transformer: FC-BN, prior scale multiplication, sparsemax
  - Sequential attention accumulates masked features across N_steps decision steps
  - Sparsity regularization: entropy of selection masks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class GhostBN(nn.Module):
    """Ghost Batch Normalization (Hoffer et al. 2017).

    Splits the batch into virtual sub-batches of size vbs and applies
    standard BN to each chunk separately, then concatenates results.
    Reduces variance in BN statistics, beneficial for large batch training.
    """

    def __init__(self, num_features: int, vbs: int = 128, momentum: float = 0.02):
        super().__init__()
        self.vbs = vbs
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= self.vbs or not self.training:
            return self.bn(x)
        chunks = x.split(self.vbs, dim=0)
        return torch.cat([self.bn(c) for c in chunks], dim=0)


def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax activation (Martins & Astudillo 2016).

    Projects the input onto the probability simplex while encouraging
    sparse (zero) assignments — the key to TabNet's interpretable
    feature selection.
    """
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    cumsum = torch.cumsum(z_sorted, dim=dim)

    n = z.shape[dim]
    k = torch.arange(1, n + 1, device=z.device, dtype=z.dtype)
    # Reshape k so it broadcasts along `dim`
    shape = [1] * z.dim()
    shape[dim] = -1
    k = k.view(shape)

    support = (k * z_sorted > cumsum - 1).float()
    k_z = support.sum(dim=dim, keepdim=True)

    # Threshold τ(z)
    k_z_idx = (k_z - 1).long().clamp(min=0)
    tau = (cumsum.gather(dim, k_z_idx) - 1.0) / k_z

    return torch.clamp(z - tau, min=0.0)


class GLULayer(nn.Module):
    """Fully-connected → Ghost BN → Gated Linear Unit block.

    The FC doubles the output width; GLU splits it in half and gates
    the first half with sigmoid of the second half.
    """

    def __init__(self, in_features: int, out_features: int, vbs: int = 128, momentum: float = 0.02):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features * 2, bias=False)
        self.bn = GhostBN(out_features * 2, vbs=vbs, momentum=momentum)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bn(self.fc(x))
        h1, h2 = h.chunk(2, dim=-1)
        return h1 * torch.sigmoid(h2)


# ---------------------------------------------------------------------------
# Feature transformer
# ---------------------------------------------------------------------------

class FeatureTransformer(nn.Module):
    """Combines shared layers with step-specific layers (Figure 4c).

    Shared layers: same weights across all N_steps decision steps.
    Step-dependent layers: unique weights per step.
    All layers after the first use a normalized residual connection * sqrt(0.5).
    """

    def __init__(
        self,
        n_features: int,
        n_out: int,           # N_d + N_a
        n_shared: int = 2,
        n_step_dep: int = 2,
        vbs: int = 128,
        momentum: float = 0.02,
        shared_layers: nn.ModuleList | None = None,
    ):
        super().__init__()
        self.SQRT_HALF = np.sqrt(0.5)

        # Shared layers are created externally so they truly share weights
        # across all FeatureTransformer instances (one per step).
        if shared_layers is not None:
            self.shared = shared_layers
        else:
            layers = []
            for i in range(n_shared):
                layers.append(GLULayer(
                    n_features if i == 0 else n_out,
                    n_out, vbs=vbs, momentum=momentum,
                ))
            self.shared = nn.ModuleList(layers)

        # Step-dependent layers (unique per step)
        step_layers = []
        for _ in range(n_step_dep):
            step_layers.append(GLULayer(n_out, n_out, vbs=vbs, momentum=momentum))
        self.step_dep = nn.ModuleList(step_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared path — first layer has no skip (D → N_d+N_a dims differ)
        h = self.shared[0](x)
        for layer in self.shared[1:]:
            h = (h + layer(h)) * self.SQRT_HALF

        # Step-dependent path — all layers have skip connections
        for layer in self.step_dep:
            h = (h + layer(h)) * self.SQRT_HALF

        return h


# ---------------------------------------------------------------------------
# Attentive transformer
# ---------------------------------------------------------------------------

class AttentiveTransformer(nn.Module):
    """Single-layer attentive transformer with prior scale (Figure 4d).

    Maps the N_a-dimensional representation from the previous step
    to D feature-selection weights, then applies sparsemax.
    """

    def __init__(self, n_a: int, n_features: int, vbs: int = 128, momentum: float = 0.02):
        super().__init__()
        self.fc = nn.Linear(n_a, n_features, bias=False)
        self.bn = GhostBN(n_features, vbs=vbs, momentum=momentum)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, h_a: torch.Tensor, prior_scale: torch.Tensor) -> torch.Tensor:
        # h_a: (B, N_a)  prior_scale: (B, D)
        a = self.bn(self.fc(h_a))
        return sparsemax(prior_scale * a, dim=-1)


# ---------------------------------------------------------------------------
# TabNet encoder and regressor
# ---------------------------------------------------------------------------

class TabNetEncoder(nn.Module):
    """TabNet encoder: sequential attention over N_steps decision steps.

    At each step:
      1. Attentive transformer selects salient features (sparse mask M[i]).
      2. Prior scale P[i] penalizes re-using features already selected.
      3. Feature transformer processes masked features → d[i] + a[i+1].
      4. ReLU(d[i]) is accumulated into the decision embedding d_out.

    Returns d_out, the sparsity entropy loss, and per-step masks.
    """

    def __init__(
        self,
        n_features: int,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        n_shared: int = 2,
        n_step_dep: int = 2,
        vbs: int = 128,
        momentum: float = 0.02,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma

        # Input BN — uses standard (not ghost) BN as paper specifies
        self.initial_bn = nn.BatchNorm1d(n_features, momentum=momentum)

        # Shared GLU layers: built once, passed to every FeatureTransformer
        n_out = n_d + n_a
        shared_fc = []
        for i in range(n_shared):
            shared_fc.append(GLULayer(
                n_features if i == 0 else n_out,
                n_out, vbs=vbs, momentum=momentum,
            ))
        shared_layers = nn.ModuleList(shared_fc)

        # One FeatureTransformer and one AttentiveTransformer per step
        self.feat_transformers = nn.ModuleList()
        self.att_transformers = nn.ModuleList()
        for _ in range(n_steps):
            self.feat_transformers.append(FeatureTransformer(
                n_features, n_out, n_shared, n_step_dep, vbs, momentum,
                shared_layers=shared_layers,
            ))
            self.att_transformers.append(
                AttentiveTransformer(n_a, n_features, vbs=vbs, momentum=momentum)
            )

        # Register the shared layers so they appear in state_dict / optimizer
        self.shared_layers = shared_layers

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        x_bn = self.initial_bn(x)

        # Initialize prior scale P[0] = 1 (no prior on feature usage)
        prior_scale = torch.ones(B, self.n_features, device=x.device, dtype=x.dtype)
        # h_a[0] = 0  (no output from a non-existent step 0)
        h_a = torch.zeros(B, self.n_a, device=x.device, dtype=x.dtype)

        d_out = torch.zeros(B, self.n_d, device=x.device, dtype=x.dtype)
        sparsity_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        masks = []

        for step in range(self.n_steps):
            # M[step] = sparsemax(P[step-1] * h_step(a[step-1]))
            M = self.att_transformers[step](h_a, prior_scale)
            masks.append(M)

            # P[step] = P[step-1] * (gamma - M[step])
            prior_scale = prior_scale * (self.gamma - M)

            # Feature transformer on masked input
            h = self.feat_transformers[step](M * x_bn)

            # Split: d[step] → decision, h_a → next attentive transformer
            d_i = F.relu(h[:, : self.n_d])
            h_a = h[:, self.n_d :]

            d_out = d_out + d_i

            # Sparsity regularization: entropy of the mask
            sparsity_loss = sparsity_loss + (-M * torch.log(M + 1e-15)).mean()

        sparsity_loss = sparsity_loss / self.n_steps
        return d_out, sparsity_loss, masks

    @torch.no_grad()
    def aggregate_importance(self, masks: list[torch.Tensor]) -> torch.Tensor:
        """Aggregate feature importance across all steps (eq. from paper).

        M_agg[b,j] = sum_i η_b[i] * M[i][b,j] / sum_j sum_i η_b[i] * M[i][b,j]
        We approximate η_b[i] = 1 (uniform step weighting) for simplicity.
        Returns a (D,) tensor with population-level importance.
        """
        stacked = torch.stack(masks, dim=1)       # (B, N_steps, D)
        agg = stacked.mean(dim=1).mean(dim=0)     # (D,)
        total = agg.sum()
        return agg / (total + 1e-15)


class TabNetRegressor(nn.Module):
    """TabNet regressor: encoder + single linear output head."""

    def __init__(
        self,
        n_features: int,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        n_shared: int = 2,
        n_step_dep: int = 2,
        vbs: int = 128,
        momentum: float = 0.02,
        lambda_sparse: float = 1e-3,
    ):
        super().__init__()
        self.lambda_sparse = lambda_sparse
        self.encoder = TabNetEncoder(
            n_features, n_d, n_a, n_steps, gamma,
            n_shared, n_step_dep, vbs, momentum,
        )
        # Paper uses W_final with no bias
        self.fc_out = nn.Linear(n_d, 1, bias=False)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, x: torch.Tensor):
        d_out, sparsity_loss, masks = self.encoder(x)
        pred = self.fc_out(d_out).squeeze(-1)
        return pred, sparsity_loss, masks

    def loss(self, pred: torch.Tensor, y: torch.Tensor, sparsity_loss: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, y) + self.lambda_sparse * sparsity_loss

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        pred, _, _ = self(x)
        return pred.cpu().numpy()

    @torch.no_grad()
    def feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Return aggregate feature importance vector (D,) for input batch x."""
        self.eval()
        _, _, masks = self(x)
        return self.encoder.aggregate_importance(masks)
