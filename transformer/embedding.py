import torch
from einops import rearrange, einsum
import torch.nn.functional as F


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: str | None = None, dtype=None):
        super().__init__()
        w = torch.zeros((num_embeddings, embedding_dim), dtype=dtype).to(device)
        torch.nn.init.trunc_normal_(w, 0, 1, -3, 3)
        self.num_embeddings = num_embeddings

        self.weight = torch.nn.Parameter(w, requires_grad=True)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(token_ids, self.num_embeddings).to(dtype=self.weight.dtype)
        return einsum(one_hot, self.weight, "... d_in, d_in d_out -> ... d_out")
