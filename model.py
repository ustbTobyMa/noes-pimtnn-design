"""Multi-task regression and a local descriptor-gradient penalty (SI Eqs. S2-S4)."""

import torch
from torch import nn
from torch.nn import functional as F

from dataset import TARGETS


class MultiTaskNet(nn.Module):
    """Shared composition/process encoder, optional descriptor branch and three heads."""

    def __init__(self, n_features, n_descriptors=0, width=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU()
        )
        self.descriptor_encoder = None
        head_width = width
        if n_descriptors:
            self.descriptor_encoder = nn.Sequential(nn.Linear(n_descriptors, width // 2), nn.SiLU())
            head_width += width // 2
        self.heads = nn.ModuleDict({
            target: nn.Sequential(nn.Linear(head_width, width // 2), nn.SiLU(), nn.Linear(width // 2, 1))
            for target in TARGETS
        })

    def forward(self, features, descriptors=None):
        shared = self.encoder(features)
        if self.descriptor_encoder is not None:
            if descriptors is None:
                raise ValueError("This model requires descriptor inputs.")
            shared = torch.cat((shared, self.descriptor_encoder(descriptors)), dim=1)
        return torch.cat([head(shared) for head in self.heads.values()], dim=1)


def observed_mse(prediction, target):
    """Average the per-task MSE over tasks with at least one observed label."""
    observed = torch.isfinite(target)
    counts = observed.sum(dim=0)
    present = counts > 0
    if not present.any():
        raise ValueError("The training batch has no target values.")
    errors = torch.where(observed, prediction - torch.nan_to_num(target), 0.0)
    task_sse = errors.square().sum(dim=0)
    return (task_sse[present] / counts[present]).mean()


def directional_penalty(prediction, descriptor_z, observed, signs):
    """Mean hinge over observed descriptor/task pairs with a non-zero prior.

    signs[k,j] is +1 for increasing, -1 for decreasing, or 0 for unspecified.
    descriptor_z is the differentiable, standardized tensor used in the forward pass.
    """
    if signs.shape != (prediction.shape[1], descriptor_z.shape[1]) or observed.shape != descriptor_z.shape:
        raise ValueError("The sign matrix or observation mask has incompatible dimensions.")
    if not ((signs == 0) | (signs == 1) | (signs == -1)).all():
        raise ValueError("Directional signs must be -1, 0 or +1.")
    total = prediction.sum() * 0.0
    comparisons = 0
    for k, target_signs in enumerate(signs):
        valid = observed & (target_signs != 0)
        if valid.any():
            gradient, = torch.autograd.grad(prediction[:, k].sum(), descriptor_z, create_graph=True)
            total = total + F.relu(-gradient * target_signs)[valid].sum()
            comparisons += int(valid.sum())
    return total / max(comparisons, 1)
