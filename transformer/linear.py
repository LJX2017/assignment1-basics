import torch
from einops import rearrange, einsum
import math


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: str | None = None, dtype=None):
        super().__init__()
        w = torch.zeros((out_features, in_features), dtype=dtype).to(device)
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(w, 0, std ** 2, -3 * std, 3 * std)

        self.weight = torch.nn.Parameter(w, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class MLP(torch.nn.Module):
    def __init__(self, n_embd: int, ff_dim: int | None = None, device: str | None = None, dtype=None) -> None:
        super().__init__()
        if not ff_dim:
            self.ff_dim = int(((n_embd * 8.0 / 3.0) // 64) * 64)
        else:
            self.ff_dim = ff_dim
        # w1 = torch.zeros((ff_dim, n_embd), dtype=dtype).to(device)
        # w2 = torch.zeros((n_embd, ff_dim), dtype=dtype).to(device)
        # w3 = torch.zeros((ff_dim, n_embd), dtype=dtype).to(device)
        # torch.nn.init.trunc_normal_(w1, 0, 2.0 / (ff_dim + n_embd))
        # torch.nn.init.trunc_normal_(w2, 0, 2.0 / (ff_dim + n_embd))
        # torch.nn.init.trunc_normal_(w3, 0, 2.0 / (ff_dim + n_embd))
        self.w1 = Linear(n_embd, self.ff_dim, device, dtype)
        self.w2 = Linear(self.ff_dim, n_embd, device, dtype)
        self.w3 = Linear(n_embd, self.ff_dim, device, dtype)

    def silu(self, x: torch.Tensor):
        return x / (1 + torch.exp(-x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.silu(self.w1(x))
        w3x = self.w3(x)
        return self.w2(w1x * w3x)
