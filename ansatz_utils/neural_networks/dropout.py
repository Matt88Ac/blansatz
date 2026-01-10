import torch


class DropoutEquivariant(torch.nn.Module):
    def __init__(self, dim: int, p: float = 0.0):
        super(DropoutEquivariant, self).__init__()
        assert 0 <= p < 1, ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.dummy = torch.nn.Parameter(torch.empty(1), requires_grad=False)
        self.mask = torch.empty(dim)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.mask.bernoulli_(1-self.p)
        self.mask /= (1 - self.p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            ones = [1]*(x.dim() - 1)
            return x * self.mask.contiguous().view(*ones, -1).to(device=x.device, dtype=x.dtype)

        return x
