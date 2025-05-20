import torch

try:
    import quimb.tensor as qtn
except ImportError:
    print("quimb not installed, partial_dm will not work")
    qtn = None
import numpy as np
from tqdm import tqdm
from typing import List
from qtorch.quant import Quantizer


# kronecker product for tensor with leading batch dimension
# adapted from numpy implementation
def kron(a: torch.Tensor, b: torch.Tensor, batchs: int):
    ndb, nda = b.ndim, a.ndim
    nd = max(ndb, nda) - 1  # suppose the first dimension is batch dimension

    if nda == 0 or ndb == 0:
        return torch.multiply(a, b)

    as_ = a.shape
    bs = b.shape

    # Equalise the shapes by prepending smaller one with 1s
    as_ = (as_[0],) + (1,) * max(0, ndb - nda) + as_[1:]
    bs = (bs[0],) + (1,) * max(0, nda - ndb) + bs[1:]

    # Insert empty dimensions
    a_arr = a.view(as_)
    b_arr = b.view(bs)

    # Compute the product
    for axis in range(1, nd * 2, 2):
        a_arr = a_arr.unsqueeze(1 + axis)
    for axis in range(0, nd * 2, 2):
        b_arr = b_arr.unsqueeze(1 + axis)

    result = torch.multiply(a_arr, b_arr)

    # Reshape back
    result = result.reshape((batchs,) + tuple(np.multiply(as_[1:], bs[1:])))

    return result


# partial density matrix for data with leading batch dimension
# using quimb
if qtn is not None:

    def partial_dm(sites_index, dl, local_dim=2, device="cuda"):
        rho_dim = local_dim ** len(sites_index)
        N = 0
        rho = torch.zeros((rho_dim, rho_dim), dtype=torch.complex128, device=device)
        for batch in tqdm(dl, desc="quimb partial dm"):

            for datum in batch[0]:
                rho += (
                    qtn.MatrixProductState(datum.squeeze().reshape(-1, 1, 1, 2))
                    .partial_trace(sites_index)
                    .to_dense()
                )
                N += 1

        ## per ora ce lo teniamo così ma si può migliorare

        return rho / N


# partial density matrix for data with leading batch dimension
# using torch and assuming data is a tensor of separable states
def sep_partial_dm(
    keep_index,
    sep_states: (
        torch.utils.data.DataLoader | torch.Tensor
    ),  # TODO: support also list of tensors with eventually different shapes
    skip_norm=False,
    device="cpu",
):
    if not isinstance(keep_index, torch.Tensor):
        keep_index = torch.tensor(keep_index, device=device, dtype=torch.int64)
    if isinstance(sep_states, torch.utils.data.DataLoader):
        discard_index = torch.ones(
            next(iter(sep_states))[0].shape[-2], dtype=torch.bool
        )
        discard_index[keep_index] = False
        rho_list = []
        for batch in tqdm(sep_states, desc="sep_partial_dm", position=1):
            batch = batch[0].to(device)
            if skip_norm:
                norm_factor = torch.eye(1, device=device)
            else:
                norm_factor = torch.prod(
                    torch.sum(batch[..., discard_index, :] ** 2, dim=-1), dim=-1
                ).squeeze()

            rhos = torch.einsum(
                "...i,...j->...ij",
                batch[..., keep_index, :].conj(),
                batch[..., keep_index, :],
            )
            rho = torch.eye(1, device=device, dtype=torch.complex128)

            for i in (
                keep_index - keep_index.min()
            ):  # strange way to index but in this way we can get the partial density matrix also for different permutations of the sites
                rho = kron(rho, rhos[..., i, :, :], batchs=batch.shape[0])

            rho_list.append(
                rho * norm_factor.view([-1] + [1] * (rho.ndim - norm_factor.ndim))
            )
        return torch.concat(rho_list, dim=0)

    elif isinstance(sep_states, qtn.TensorNetwork):
        batch = [tensor.data for tensor in sep_states]
        batch = torch.stack(
            batch, dim=-2
        )  # suppose the sites dimension is the second to last
        return sep_partial_dm(keep_index, batch, skip_norm=skip_norm, device=device)

    elif isinstance(sep_states, torch.Tensor):
        # sep_states is a tensor of separable states with shape B x N x d
        # (B is optional batch dimension, N is number of sites, d is local dimension)
        batch = sep_states.to(device)
        if skip_norm:
            norm_factor = torch.eye(1, device=device, dtype=torch.complex128)
        else:
            discard_index = torch.ones(sep_states.shape[-2], dtype=torch.bool)
            discard_index[keep_index] = False
            norm_factor = torch.prod(
                torch.sum(batch[..., discard_index, :] ** 2, dim=-1), dim=-1
            ).squeeze()

        # get rhos of single sites
        rhos = torch.einsum(
            "...i,...j->...ij",
            batch[..., keep_index, :].conj(),
            batch[..., keep_index, :],
        )
        rho = torch.eye(1, device=device)

        # tensor product of rhos of single sites if they are selected
        for i in keep_index - keep_index.min():
            rho = kron(rho, rhos[..., i, :, :], batchs=batch.shape[0])

        return rho * norm_factor.view([-1] + [1] * (rho.ndim - norm_factor.ndim))

    else:
        raise TypeError(
            f"sep_states must be one of torch.utils.data.DataLoader, torch.Tensor or quimb.tensor.TensorNetwork, got: {type(sep_states)}"
        )


# partial density matrix for data with leading batch dimension
# using torch ONLY and assuming data is a tensor of separable states
def sep_partial_dm_torch(
    keep_index,
    sep_states: (
        torch.utils.data.DataLoader | torch.Tensor
    ),  # TODO: support also list of tensors with eventually different shapes
    skip_norm=False,
    device="cpu",
):
    if not isinstance(keep_index, torch.Tensor):
        keep_index = torch.tensor(keep_index, device=device, dtype=torch.int64)
    if isinstance(sep_states, torch.utils.data.DataLoader):
        discard_index = torch.ones(
            next(iter(sep_states))[0].shape[-2], dtype=torch.bool
        )
        discard_index[keep_index] = False
        rho_list = []
        for batch in tqdm(sep_states, desc="sep_partial_dm", position=1):
            batch = batch[0].to(device)
            if skip_norm:
                norm_factor = torch.eye(1, device=device)
            else:
                norm_factor = torch.prod(
                    torch.sum(batch[..., discard_index, :] ** 2, dim=-1), dim=-1
                ).squeeze()

            rhos = torch.einsum(
                "...i,...j->...ij",
                batch[..., keep_index, :].conj(),
                batch[..., keep_index, :],
            )
            rho = torch.eye(1, device=device, dtype=torch.complex128)

            for i in (
                keep_index - keep_index.min()
            ):  # strange way to index but in this way we can get the partial density matrix also for different permutations of the sites
                rho = kron(rho, rhos[..., i, :, :], batchs=batch.shape[0])

            rho_list.append(
                rho * norm_factor.view([-1] + [1] * (rho.ndim - norm_factor.ndim))
            )
        return torch.concat(rho_list, dim=0)

    elif isinstance(sep_states, torch.Tensor):
        # sep_states is a tensor of separable states with shape B x N x d
        # (B is optional batch dimension, N is number of sites, d is local dimension)
        batch = sep_states.to(device)
        if skip_norm:
            norm_factor = torch.eye(1, device=device, dtype=sep_states.dtype)
        else:
            discard_index = torch.ones(sep_states.shape[-2], dtype=torch.bool)
            discard_index[keep_index] = False
            norm_factor = torch.prod(
                torch.sum(batch[..., discard_index, :] ** 2, dim=-1), dim=-1
            ).squeeze()

        # get rhos of single sites
        rhos = torch.einsum(
            "...i,...j->...ij",
            batch[..., keep_index, :].conj(),
            batch[..., keep_index, :],
        )
        rho = torch.eye(1, device=device)

        # tensor product of rhos of single sites if they are selected
        for i in keep_index - keep_index.min():
            rho = kron(rho, rhos[..., i, :, :], batchs=batch.shape[0])

        return rho * norm_factor.view([-1] + [1] * (rho.ndim - norm_factor.ndim))

    else:
        raise TypeError(
            f"sep_states must be one of torch.utils.data.DataLoader, torch.Tensor or quimb.tensor.TensorNetwork, got: {type(sep_states)}"
        )


# perform network contraction assuming
# data is a tn representing separable states
# and tensor_list is a list of tensors to contract
if qtn is not None:

    def sep_contract(tensor_list, data_tn: qtn.TensorNetwork):

        results = []
        for tensor in tensor_list:
            contr = tensor
            for ind in tensor.inds[:2]:
                contr = (contr & data_tn._select_tids(data_tn.ind_map[ind])).contract(
                    ..., output_inds=["b", tensor.inds[-1]]
                )
            results.append(contr)
        return qtn.TensorNetwork(results)


# perform network contraction assuming each
# tensor in tensor_list has to be contracted
# with two contiguous tensors in data_tensors
def sep_contract_torch(tensors, data_tensors):

    is_batch = data_tensors[0].ndim > 1
    if is_batch:
        contr_string = "xyz,bx,by->bz"
    else:
        contr_string = "xyz,x,y->z"

    results = []
    for i, tensor in enumerate(tensors):
        results.append(
            torch.einsum(
                contr_string, tensor, data_tensors[2 * i], data_tensors[2 * i + 1]
            )
        )

    return torch.stack(results).squeeze()


# @torch.jit.script
def contract_up(
    tensor: torch.Tensor,
    data_tensors: List[torch.Tensor],
    quantizer: Quantizer | None = None,
):
    """
    This function is the fundamental block for TTN contractions.
    It contracts a tensor with two contiguous data tensors.

    Args:
    - tensor: torch.Tensor, the tensor to contract
    - data_tensors: List[torch.Tensor], the two data vectors to contract with the tensor

    Returns:
    - right: torch.Tensor, the contracted tensor
    """

    if quantizer is not None:
        tensor = quantizer(tensor)
        left = quantizer(
            torch.matmul(
                quantizer(data_tensors[0]),
                tensor.contiguous().view(tensor.shape[0], -1),
            )
        )
        right = quantizer(
            torch.bmm(
                quantizer(data_tensors[1]).unsqueeze(1),
                left.view(-1, tensor.shape[1], tensor.shape[2]),
            )
        )
    else:
        # left contraction with data (b x p) @ (p x d) -> (b x d) where d is m*n
        left = torch.matmul(
            data_tensors[0], tensor.contiguous().view(tensor.shape[0], -1)
        )
        # right contraction with data (b x 1 x m) @ (b x m x n) -> (b x n)
        right = torch.bmm(
            data_tensors[1].unsqueeze(1),
            left.view(-1, tensor.shape[1], tensor.shape[2]),
        )

    return right.view(-1, tensor.shape[2])
