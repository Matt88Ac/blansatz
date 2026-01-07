import torch
from torch import autograd


class AllDifferences(autograd.Function):
    """ This class implements both a function that maps a tensor X to [X_j - X_i: 1 <= i < j <= len(X.T)]
        and it's gradient.
    """

    @staticmethod
    def forward(ctx, inp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ctx: Neglect.
            inp (torch.Tensor):
                A real tensor X of shape [b, d_1, ..., d_k, n]

        Returns:
            A real tensor, of shape [b, d_1, ..., d_k, n(n-1)/2], for [X_j - X_i: 1 <= i < j <= n]
        """
        *b, n = inp.shape
        differences = []
        for i in range(n - 1):
            differences.append(
                inp[..., i + 1:] - inp[..., i:i + 1]
            )
        ctx.save_for_backward(inp)
        ctx.n = n
        ctx.size = n * (n - 1) // 2
        ctx.b = b
        return torch.cat(differences, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        X, = ctx.saved_tensors
        n = ctx.n
        b = ctx.b
        size = ctx.size

        grad_input = torch.zeros(*b, n, size, device=X.device, dtype=X.dtype)
        last_sub_matrix_size = 0
        sub_matrix_size = n - 1
        to_add = n - 2
        for i in range(n):
            grad_input[..., i, last_sub_matrix_size:sub_matrix_size] += -1
            grad_input[..., i + 1:, last_sub_matrix_size:sub_matrix_size] += torch.eye(to_add + 1, dtype=X.dtype,
                                                                                       device=X.device)
            last_sub_matrix_size = sub_matrix_size
            sub_matrix_size += to_add
            to_add -= 1

        grad_input = (grad_input * grad_output.unsqueeze(len(b))).sum(dim=-1)

        return grad_input


def vandermonde_determinant(X: torch.Tensor) -> torch.Tensor:
    """
    Args:
        X (torch.Tensor):
            A real tensor X of shape [b, d_1, ..., d_k, n]

    Returns:
        The Vandermonde Determinant of X along the last axis.
    """
    return AllDifferences.apply(X).prod(-1, True)


def uniform_sphere_sampling(n_samples: int, dim: int) -> torch.Tensor:
    samples = torch.randn(n_samples, dim)
    return samples / torch.norm(samples, dim=-1, keepdim=True)


# @torch.jit.script
# @torch.compile(fullgraph=True, dynamic=True)
def linear_wsop_sub_weights(X: torch.Tensor) -> torch.Tensor:
    """
        Args:
            X (torch.Tensor):
                A real tensor X of shape [b, d_1, ..., d_k, n]
            offset (float):
                A real number that offsets the sub-weights
            scale (float):
                A real number that scales the sub-weights
        Returns (torch.Tensor):
                A real tensor of shape [b, d_1, ..., d_k, n],
                where on the last axis, at each element i, scale * (1/x_i) / (offset + sum(1/x_j, 1 <= j <= n))
            """

    is_zero = torch.isclose(X, torch.zeros_like(X))

    zero_count = is_zero.sum(dim=-1, keepdim=True)
    zero_count = torch.where(zero_count > 0, 1 / zero_count, zero_count) + torch.zeros_like(X)
    # zero_count = is_zero * zero_count

    out = torch.where(
        torch.logical_and(zero_count > 0, is_zero > 0),
        zero_count,
        torch.zeros_like(X)
    )

    out = torch.where(
        zero_count == 0,
        (1 / X) / (1 + torch.sum(1 / X, dim=-1, keepdim=True)),
        out
    ).nan_to_num(0.0, 0.0, 0.0) / 2

    return out


if __name__ == '__main__':
    from permutation_actions import all_transpositions


    def make_transpositions(_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        *size, d, n = _x.shape
        at = all_transpositions(_x.shape[-2], _x.device).unsqueeze(0).unsqueeze(-1)
        for i in range(len(size) - 1):
            at = at.unsqueeze(2)

        shaped_input = _x.clone().unsqueeze(1)
        trans_input = torch.take_along_dim(shaped_input, at, -2)
        # print(shaped_input.shape, trans_input.shape, _x.shape)
        return shaped_input, trans_input


    x = torch.rand(1, 5, 30, device='cuda')
    x[:, :, 3] = x[:, :, 2]

    xs, xt = make_transpositions(x.transpose(-2, -1))

    xx = (xs - xt).norm(dim=(-2, -1))

    print(linear_wsop_sub_weights(xx))

    # b, k, n = 10, 8, 13
    # temp = torch.rand(b, k, n, device='cpu', dtype=torch.float64, requires_grad=True) * 100 + 7
