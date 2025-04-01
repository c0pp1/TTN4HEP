import torch

__all__ = ["embeddings_dict"]


# quantum embedding
def spin_map(tensor, dim=2):  # TODO: extend to higher dimensions
    cos = torch.cos(torch.pi * tensor)
    sin = torch.sin(torch.pi * tensor)

    assert torch.allclose(cos**2 + sin**2, torch.ones_like(cos))
    return torch.stack([cos, sin], dim=-1)


def poly_map(tensor, dim=2):

    powers = (
        torch.stack([tensor**i for i in range(dim)], dim=-1) + 1e-15
    )  # add a small number to avoid division by zero
    result = powers / torch.linalg.vector_norm(powers, dim=-1, keepdim=True)
    assert torch.allclose(
        torch.sum(result**2, dim=-1), torch.ones_like(tensor, dtype=result.dtype)
    )
    return result


def stacked_map(tensor: torch.Tensor, dim=None):
    return tensor / torch.linalg.vector_norm(tensor, dim=-1, keepdim=True)


def stacked_poly_map(tensor: torch.Tensor, dim=2):
    powers = (
        torch.stack([tensor**i for i in range(dim)], dim=-1) + 1e-15
    )  # add a small number to avoid division by zero
    result = powers / torch.linalg.vector_norm(powers, dim=-1, keepdim=True)
    assert torch.allclose(
        torch.sum(result**2, dim=-1), torch.ones_like(tensor, dtype=result.dtype)
    )
    return result


embeddings_dict = {"spin": spin_map, "poly": poly_map}
