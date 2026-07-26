import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    return F.scaled_dot_product_attention(Q, K, V)
