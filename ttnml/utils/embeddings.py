import torch
from functools import partial

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
    if tensor[0].ndim != 2:
        raise ValueError("Each tensor should have 2 dimensions")

    result = torch.concatenate(
        [
            torch.ones(list(tensor.shape[:2]) + [1]),
            torch.stack([tensor ** (i + 1) for i in range(dim)], axis=-1).reshape(
                *tensor.shape[:2], -1
            ),
        ],
        axis=-1,
    )

    return (result / torch.linalg.norm(result, axis=-1, keepdims=True))[
        0 if len(tensor) == 1 else ...
    ]


def interaction_mapping(tensor: torch.Tensor, dim=2, feat_map: callable = spin_map):
    # here we suppose that the tensor is a 3D tensor [B, N, F] where B is the
    # batch size, N is the number of sites and F is the number of features.
    # We will map the tensor to a 2D tensor [B, N*F] assuming that the first
    # half of the features is of type 1 and the second half is of type 2.
    # (geometrical / physical features) and reordering them s.t. the first half
    # goes into the left half of the system and the second half goes into the
    # right half of the system.

    if len(tensor) == 1:
        tensor = tensor[0]
        N, F = tensor.shape
        B = 1
    else:
        B, N, F = tensor.shape

    assert F % 2 == 0, f"The number of features should be even. Got {F}"
    F = F // 2
    tensor = tensor.reshape(B, N, 2, F)

    left = feat_map(tensor[:, :, 0], dim)
    right = feat_map(tensor[:, :, 1], dim)

    total = torch.concatenate([left, right], axis=-2)
    return total


def the_stacked_mapping(tensor: torch.Tensor, n_part_per_site, part_per_feat, dim=2):
    """
    This mapping function takes a tensor of shape (B, N, F) and maps it in the following way:
    1. unstack the input in F tensors of shape (B, N)
    2. for each feature f in F, take the particles indeces indicated by part_per_feat[f].
       N_f = len(part_per_feat[f]) must be a multiple of n_part_per_site
    3. for each feature f, map the tensor of shape (B, N_f) to a tensor of shape
       (B, N_f/n_part_per_site, n_part_per_site * dim + 1), using the stacked_poly_map function
    4. concatenate the F tensors along the axis before the last one
    5. return the tensor of shape (B, \sum_f N_f/n_part_per_site, n_part_per_site * dim + 1)


    Args:
        tensor (np.ndarray): input tensor of shape (B, N, F)
        n_part_per_site (int): number of particles per site
        n_part_per_feat (list[np.ndarray | list]): list of slices indicating the particles indices for each feature
        dim (int, optional): local dimension. Defaults to 2.
    """

    if len(tensor) == 1:
        tensor = tensor[0]
        B, N, F = 1, *tensor.shape
    else:
        B, N, F = tensor.shape

    assert all(
        len(part_per_feat_f) % n_part_per_site == 0 for part_per_feat_f in part_per_feat
    )

    tensors = []
    for f in range(F):
        part_indices = part_per_feat[f]
        N_f = len(part_indices)
        N_f //= n_part_per_site
        tensors.append(
            stacked_poly_map(
                tensor[:, part_indices, f].reshape(B, N_f, n_part_per_site), dim=dim
            )
        )

    return torch.concatenate(tensors, axis=-2)


embeddings_dict = {
    "spin": spin_map,
    "poly": poly_map,
    "stacked": stacked_map,
    "stacked_poly": stacked_poly_map,
    "interaction_spin": partial(interaction_mapping, feat_map=spin_map),
    "interaction_poly": partial(interaction_mapping, feat_map=poly_map),
    "interaction_stacked": partial(interaction_mapping, feat_map=stacked_map),
    "interaction_stacked_poly": partial(interaction_mapping, feat_map=stacked_poly_map),
    "the_stacked_mapping": the_stacked_mapping,
}
