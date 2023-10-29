import torch
import torchvision as tv
from quimb import tensor as qtn
import numpy as np
from tqdm.autonotebook import tqdm
from functools import partial
from typing import Dict

from algebra import sep_contract, sep_partial_dm


############# TTN #############
###############################
class TTN(qtn.TensorNetwork):
    def __init__(
        self,
        n_features,
        n_phys=2,
        n_labels=2,
        label_tag="label",
        bond_dim=4,
        virtual=False,
        device="cpu",
    ):
        if (n_features % 2) != 0:
            raise ValueError(f"n_features must be  power of 2, got: {n_features}")

        self.n_features = n_features
        self.n_phys = n_phys
        self.n_layers = int(np.log2(n_features))
        self.n_labels = n_labels
        self.label_tag = label_tag
        self.bond_dim = bond_dim
        self.virtual = virtual
        self.device = device
        self.initialized = False
        self.id = np.zeros([self.bond_dim] * 3)
        for i in range(self.id.shape[0]):
            self.id[i, i, i] = 1.0

        # add first tensor with special index
        if not (self.n_layers - 1):
            return qtn.TensorNetwork(
                [
                    qtn.rand_tensor(
                        shape=(self.n_phys, self.n_phys, self.n_labels),
                        inds=["00.000", "00.001", self.label_tag],
                        tags=["l0"],
                        dtype="complex128",
                    )
                ]
            )
        else:
            dim = min(self.n_phys ** (self.n_layers), self.bond_dim)
            tensors = [
                qtn.rand_tensor(
                    shape=(dim, dim, self.n_labels),
                    inds=["00.000", "00.001", self.label_tag],
                    tags=["l0"],
                    dtype="complex128",
                )
            ]

        for l in range(1, self.n_layers - 1):  # constructing the ttn starting from the top
            dim_pre = min(self.n_phys ** (self.n_layers - l), self.bond_dim)
            dim_post = min(self.n_phys ** (self.n_layers - l + 1), self.bond_dim)
            tensors.extend(
                [
                    qtn.rand_tensor(
                        shape=[dim_pre] * 2 + [dim_post],
                        inds=[
                            f"{l:02}.{2*i:03}",
                            f"{l:02}.{2*i+1:03}",
                            f"{(l-1):02}.{i:03}",
                        ],
                        tags=[f"l{l}"],
                        dtype="complex128",
                    )
                    if np.random.rand() < 0.5
                    else qtn.Tensor(
                        data=self.id[:dim_pre, :dim_pre, :dim_post],
                        inds=[
                            f"{l:02}.{2*i:03}",
                            f"{l:02}.{2*i+1:03}",
                            f"{(l-1):02}.{i:03}",
                        ],
                        tags=[f"l{l}"],
                    )
                    for i in range(2**l)
                ]
            )

        dim = min(self.n_phys**2, self.bond_dim)
        tensors.extend(
            [
                qtn.rand_tensor(
                    shape=[self.n_phys] * 2 + [dim],
                    inds=[
                        f"{self.n_layers-1:02}.{2*i:03}",
                        f"{self.n_layers-1:02}.{2*i+1:03}",
                        f"{(self.n_layers-2):02}.{i:03}",
                    ],
                    tags=[f"l{self.n_layers-1}"],
                    dtype="complex128",
                )
                for i in range(2 ** (self.n_layers - 1))
            ]
        )

        super(TTN, self).__init__(tensors, virtual=virtual)
        self.apply_to_arrays(
            lambda x: torch.tensor(x, device=device, dtype=torch.complex128)
        )

    def draw(self, return_fig=False, ax=None):
        fig = super().draw(
            color=self.tags,
            show_inds="all",
            fix=dict(
                {
                    key: (4 * i / self.n_features - 1.0, -1.0)
                    for i, key in enumerate(
                        list(self.tensor_map.keys())[: -self.n_features // 2 - 1 : -1]
                    )
                },
                **{"l0": (0.0, 1.0)},
            ),
            return_fig=return_fig,
            ax=ax,
            dim=2,
            figsize=(15, 10),
        )
        return fig

    def copy(self):
        return TTN(
            self.n_features,
            self.n_phys,
            self.n_labels,
            self.label_tag,
            self.bond_dim,
            self.virtual,
            self.device,
        )

    def initialize(self, train_dl: torch.utils.data.DataLoader):
        ttn_init(self, train_dl, device=self.device)
        self.initialized = True


class TNModel(torch.nn.Module):
    def __init__(self, myttn: TTN, init: Dict, batched_forward=False):
        super().__init__()
        
        self.myttn = myttn
        # we can choose wether or not to apply the initialization
        # proposed in https://doi.org/10.1088/2058-9565/aaba1a by setting init['initialize'] to True.
        # if we want to initialize the model, we need to pass a dataloader to the init dict
        if init['initialize'] and not self.myttn.initialized:
            self.myttn.initialize(init['dataloader'])

        self.batched_forward = batched_forward # if true, forward function will be faster but will require more memory
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(self.myttn)

        # convert quimb tensors to torch parameters
        self.torch_params = torch.nn.ParameterDict(
            {
                # torch requires strings as keys
                str(i): torch.nn.Parameter(initial)
                for i, initial in params.items()
            }
        )


    def forward(self, x: torch.Tensor):
        # convert back to original int key format
        params = {int(i): p for i, p in self.torch_params.items()}
        # reconstruct the TN with the new parameters
        tn = qtn.unpack(params, self.skeleton)

        if self.batched_forward:
            # construct data tn
            data_tn = qtn.TensorNetwork(
                [
                    qtn.Tensor(
                        data=site_batch,
                        inds=["b", f"{self.myttn.n_layers-1:02}.{i:03}"],
                        tags="data",
                    )
                    for i, site_batch in enumerate(torch.unbind(x, -2))
                ]
            )

            # contract data tn with model tn
            result = (
                (tn & data_tn)
                .contract(tags=..., output_inds=["b", self.myttn.label_tag])
                .data
            )
            del data_tn
            return result
        else:
            results = []

            for datum in x:
                # adapt datum to mps
                for _ in range(4 - datum.dim()):
                    datum = datum.unsqueeze(0)
                contr = (
                    qtn.MatrixProductState(
                        torch.unbind(datum, -2),
                        site_ind_id=f"{self.myttn.n_layers-1:02}.{{:03}}",
                    )
                    & tn
                ) ^ ...
                results.append(contr.data)

            return torch.stack(results)
    
    def draw(self, return_fig=False):
        fig = self.myttn.draw(return_fig=return_fig)
        return fig


########### TTN UTILS ###########
#################################

def ttn_init(ttn: TTN, train_dl: torch.utils.data.DataLoader, device="cpu"):
    # now we want to run across the ttn, layer by layer
    # and initialize the tensors by getting the partial dm
    # of two sites of the previous layer, diagonalizing it,
    # and isometrizing the rotation matrix (with n eigenvectors
    # corresponding to the n=bond_dim greatest eigenvalues)
    data_tn_batched = []
    for batch in tqdm(train_dl, desc="preparing dataset"):
        batch = batch[0].squeeze().to(device=device, dtype=torch.complex128)
        data_quimb = [
            qtn.Tensor(
                data=site_batch, inds=["b", f"{ttn.n_layers-1:02}.{i:03}"], tags="data"
            )
            for i, site_batch in enumerate(torch.unbind(batch, -2))
        ]
        data_tn = qtn.TensorNetwork(data_quimb)
        data_tn_batched.append(data_tn)

    pbar = tqdm(
        total=(ttn.n_layers - 1) * len(data_tn_batched)
        + 2 * (2 ** (ttn.n_layers - 1) - 1),
        desc="ttn init",
        position=1,
        leave=True,
    )
    for layer in range(ttn.n_layers - 1, 0, -1):  # do this for all layers except the uppermost one
        pbar.set_postfix_str(f"doing layer {layer}")
        next_layer_list = []
        ttn_curr_layer = ttn.select_tensors(f"l{layer}")
        # perform initialization of current layer with partial dm
        # of state at previous layer
        for i, tensor in enumerate(ttn_curr_layer):
            pbar.set_postfix_str(f"doing layer {layer}, tensor {i+1}/{2**layer}")
            sel_sites = [int(index.split(".")[-1]) for index in tensor.inds[:2]]
            partial_dm = torch.concat(
                [
                    sep_partial_dm(sel_sites, datum_tn, skip_norm=True, device=device)
                    for datum_tn in data_tn_batched
                ]
            ).mean(dim=0)
            eigenvectors = torch.linalg.eigh(partial_dm)[1][:, -tensor.shape[-1]:]  
            # the physical indices of the two sites are fused in the first index, we have to reshape
            tensor.modify(
                data=eigenvectors.reshape(tensor.shape),
                inds=tensor.inds,
                tags=tensor.tags,
            )
            tensor.isometrize(
                left_inds=tensor.inds[2:], method="householder", inplace=True
            )  # this operation moves the bond link to the left, we have to move it back
            tensor.moveindex(tensor.inds[0], -1, inplace=True)

            pbar.update(1)

        # calculate next propagation of data to this layer
        # with the updated tensors
        pbar.set_postfix_str(f"doing layer {layer}, propagating data")
        for data_tn in data_tn_batched:
            new_data_layer = sep_contract(ttn_curr_layer, data_tn)
            next_layer_list.append(new_data_layer)

            pbar.update(1)
        del data_tn_batched
        data_tn_batched = next_layer_list
    pbar.set_postfix_str(f'done!')
    pbar.close()

def check_correct_init(model: TNModel):
    # gives true if correctly initialized and also the number of errors
    result_list = []
    for layer in range(model.myttn.n_layers-1,0,-1):
        layer_tensors = model.myttn.select_tensors(f'l{layer}')
        for id, tensor in enumerate(layer_tensors):
            contr = tensor.contract(tensor.H, output_inds=[f'{layer-1:02}.{id:003}'])
            result = torch.allclose(contr.data, torch.ones(contr.shape[-1], dtype=torch.complex128, device='cuda'))
            if not result:
                print(f'Layer {layer}, tensor {id} is not initialized correctly')
            result_list.append(not result)
    
    n_errors = torch.tensor(result_list, dtype=torch.bool).sum().item()
    return n_errors == 0, n_errors