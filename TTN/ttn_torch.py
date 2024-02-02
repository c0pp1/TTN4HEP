from __future__     import annotations
from collections    import UserString
from typing         import Sequence, List, Tuple, ValuesView
from string         import ascii_letters
from datetime       import datetime

import colorsys
import torch
from qtorch.quant import Quantizer
import numpy as np
import graphviz

from tqdm import tqdm, trange
from algebra import contract_up, sep_partial_dm_torch
from utils import adjust_lightness, one_epoch_one_tensor_torch
from matplotlib import colors, colormaps


########## CLASSES FOR TENSOR INDEXING ##########
#################################################

# class which represent a generic tensor index,
# it is used to identify a tensor in the TTN.
# It is composed by a name and a list of indices
class TIndex:
    def __init__(self, name, inds: Sequence[str] | np.ndarray):
        self.__name = name
        self.__tindices = np.array(inds, dtype=np.str_) # problems with string lenghts
        self.__ndims = len(inds)
    
    def __getitem__(self, key: int) -> str:
        return self.__tindices[key]
    
    def __setitem__(self, key: int, value: str):
        old_len = self.__tindices.dtype.itemsize / 4
        new_len = max(old_len, len(value))
        self.__tindices = self.__tindices.astype(f"<U{new_len:.0f}")
        self.__tindices[key] = value
    
    @property
    def name(self):
        return self.__name 

    @property
    def indices(self):
        return self.__tindices
    
    @property
    def ndims(self):
        return self.__ndims
    
    ''' I do not want them to be changed by design   
    @indices.setter
    def indices(self, value: Sequence[str]):
        self.__indices = value
    '''
    
    def __eq__(self, __value: TIndex | str) -> bool:
        if isinstance(__value, str):
            return self.__name == __value
        return self.__name == __value.name and np.all(self.__tindices == __value.indices)
    
    def __gt__(self, __value: TIndex | str) -> bool:
        compare = __value if isinstance(__value, str) else __value.name
        try:    compare_layer = int(compare.split('.')[0])
        except: compare_layer = np.inf
        try:    self_layer = int(self.__name.split('.')[0])
        except: self_layer = np.inf

        if self_layer > compare_layer:
            return True
        elif self_layer == compare_layer:
            return int(self.__name.split('.')[1]) > int(compare.split('.')[1])
        return False
    
    def __lt__(self, __value: TIndex | str) -> bool:
        compare = __value if isinstance(__value, str) else __value.name
        try:    compare_layer = int(compare.split('.')[0])
        except: compare_layer = np.inf
        try:    self_layer = int(self.__name.split('.')[0])
        except: self_layer = np.inf

        if self_layer < compare_layer:
            return True
        elif self_layer == compare_layer:
            return int(self.__name.split('.')[1]) < int(compare.split('.')[1])
        return False
    
    def __ge__(self, __value: TIndex | str) -> bool:
        return self.__gt__(__value) or self.__eq__(__value)
    
    def __le__(self, __value: TIndex | str) -> bool:
        return self.__lt__(__value) or self.__eq__(__value)
    
    def __hash__(self):
        return hash(self.__name)
    
    def __str__(self) -> str:
        return self.__name
    
    def __repr__(self) -> str:
        return 'TIndex: ' + self.__name
    
    def _repr_markdown_(self):
        return f'**{self.__repr__()}**'
    
    def _repr_html_(self):
        markdown_str = f'<details><summary><b style="color:#0088d9; font-size:100%; font-family: verdana, sans-serif">{self.__repr__()} </b></summary>'
        for index in self.__tindices:
            markdown_str += f'&emsp;&ensp; <b style="color:#be00d9">{index}</b><br>'
        return markdown_str + '</details>'
    

# class which represent a tensor index in the
# specific case of the Tree Tensor Network.
# It is created by an int representing the layer
# and an int representing the index in the layer
class TTNIndex(TIndex):
    def __init__(self, layer: int, layer_index: int):

        self.__layer = layer
        self.__layer_index = layer_index
        super(TTNIndex, self).__init__(f"{layer}.{layer_index}",
                                       [f"{layer+1}.{2*layer_index}", f"{layer+1}.{2*layer_index+1}", f"{layer}.{layer_index}"], 
                                       )
    
    def __repr__(self) -> str:
        return f"TTNIndex: {self.__layer}.{self.__layer_index}"


# MEMO: THIS CLASS IS NOT USED  
# class which represent the link between two tensors in the TTN
# it is composed by a source and a target tensor index, a dimension,
# a dependencies list and a name
class TLink:
    def __init__(self, source: TIndex, target: TIndex, dim: int, dependencies: List[TIndex] = [], name: str = None):
        self.__source = source
        self.__target = target
        self.__dim = dim
        self.__vector = None
        self.__has_updated_vector = False
        self.__name = name if name is not None else f"{source.name}"
        self.__dependencies = dependencies

    @property
    def source(self):
        return self.__source
    
    @property
    def target(self):
        return self.__target
    
    @property
    def name(self):
        return self.__name
    
    @property
    def vector(self):
        if self.__has_updated_vector:
            return self.__vector
        elif self.__vector is not None:
            raise ValueError(f"No vector is set for TLink {self.__name}.")
        else:
            raise ValueError(f"TLink {self.__name} vector is not updated.")
        
    @vector.setter
    def vector(self, value):
        self.__vector = value
        self.__has_updated_vector = True

    @property
    def is_updated(self):
        return self.__has_updated_vector
    
    def depends_on(self, __value: TIndex | str) -> bool:
        return __value in self.__dependencies


########## CLASSES FOR TTN ##########
#####################################

# class representing a Tree Tensor Network.
# Here we define its structure, relying upon a
# simple dictionary of torch tensors and TTNIndices.
# Some useful methods are defined to access the tensors,
# initialize them and propagate data through the network.
class TTN:
    def __init__(
        self,
        n_features,
        n_phys=2,
        n_labels=2,
        label_tag="label",
        bond_dim=4,
        dtype=torch.cdouble,
        device="cpu",
        quantizer = None
    ):
        if (n_features % 2) != 0:
            raise ValueError(f"n_features must be  power of 2, got: {n_features}")

        self.n_features = n_features
        self.n_phys     = n_phys
        self.n_labels   = n_labels
        self.label_tag  = label_tag
        self.bond_dim   = bond_dim
        self.device     = device

        self.quantizer  = quantizer

        self.__dtype    = dtype
        self.__n_layers = int(np.log2(n_features))
        self.__tensors  = []
        self.__indices  = [TTNIndex(l, i) for l in range(self.__n_layers) for i in range(2**l)]
        # label top edge as label
        self.__indices[0][2] = label_tag
        # label bottom edges as data
        for ttnindex in self.__indices[-2**(self.__n_layers-1):]:
            ttnindex[0] = f'data.{ttnindex[0].split(".")[1]}'
            ttnindex[1] = f'data.{ttnindex[1].split(".")[1]}'
        # convert to numpy array for easier indexing
        self.__indices = np.asarray(self.__indices)

        self.__initialized = False

        ## INITIALIZE TENSORS ##
        # add first tensor with special index
        if not (self.__n_layers - 1):
            self.__tensors.append(
                torch.rand(
                    size=(self.n_phys, self.n_phys, self.n_labels),
                    dtype=self.__dtype,
                    device=self.device
                )
            )
        else:
            dim = min(self.n_phys**2**(self.__n_layers-1), self.bond_dim)
            self.__tensors.append(
                torch.rand(
                    size=(dim, dim, self.n_labels),
                    dtype=self.__dtype,
                    device=self.device
                )
            )

        for l in range(1, self.__n_layers - 1):  # constructing the ttn starting from the top
            dim_pre = min(self.n_phys**2**(self.__n_layers - l - 1), self.bond_dim)
            dim_post = min(self.n_phys **2** (self.__n_layers - l ), self.bond_dim)
            self.__tensors.extend(
                [
                    torch.rand(
                        size=[dim_pre] * 2 + [dim_post],
                        dtype=self.__dtype,
                        device=self.device
                    )
                    if np.random.rand() < 0.5
                    else torch.eye(
                        dim_pre**2, 
                        dtype=self.__dtype,
                        device=self.device)
                        .reshape(dim_pre, dim_pre, -1)[:, :, :dim_post]
                    for i in range(2**l)
                ]
            )

        dim = min(self.n_phys**2, self.bond_dim)
        self.__tensors.extend(
            [
                torch.rand(
                    size=[self.n_phys] * 2 + [dim],
                    dtype=self.__dtype,
                    device=self.device
                )
                for i in range(2 ** (self.__n_layers - 1))
            ]
        )
        ########################
        self.__tensor_map = dict(zip(self.__indices, self.__tensors))
    
    def __getitem__(self, key: Sequence[TTNIndex | str] | str | int | slice) -> dict[TTNIndex, torch.Tensor]:

        if isinstance(key, int):
            return {self.__indices[key]: self.__tensor_map[self.__indices[key]]}
        elif isinstance(key, str):
            return {self.__indices[self.__indices==key].item(): self.__tensor_map[key]}
        elif isinstance(key, Sequence):
            return {k if isinstance(k, TTNIndex) else self.__indices[self.__indices==k].item(): self.__tensor_map[k] for k in key}
        elif isinstance(key, slice):
            return {k: self.__tensor_map[k] for k in self.__indices[key]}
        else:
            raise TypeError(f"Invalid argument type: {type(key)}")
        
    @property
    def indices(self):
        return self.__indices
    
    @property
    def tensors(self):
        return self.__tensors
    
    @tensors.setter
    def tensors(self, value: Sequence[torch.Tensor] | torch.nn.ParameterList):
        self.__tensor_map = dict(zip(self.__indices, value))    # had to put this line before because the setter stops the execution of the rest of the function
        self.__tensors = value

    @property
    def dtype(self):
        return self.__dtype
    
    @property
    def n_layers(self):
        return self.__n_layers
    
    @property
    def initialized(self):
        return self.__initialized
        
    def __repr__(self) -> str:
        return f"TTN"
    
    def _repr_html_(self):
        markdown_str = f'<details><summary><b style="color:#d95100; font-size:100%; font-family: verdana, sans-serif">{self.__repr__()} </b></summary>'
        for tindex in self.__indices:
            markdown_str += f'{tindex._repr_html_()}'
        return markdown_str + '</details>'
    
    def get_branch(self, tindex: TTNIndex | str, till: str='data') -> dict[TTNIndex, torch.Tensor]:
        """
        Returns a dictionary of tensors and indices of the branch starting at tindex, going down to the bottom of the TTN.
        """
        if isinstance(tindex, str):
            tindex = self.__indices[self.__indices==tindex].item()
        branch_indices = [tindex]
        branch_layer   = [tindex]
        while till not in branch_layer[0][0]:
            branch_layer = [self.__indices[self.__indices==tindex[i]].item() for tindex in branch_layer for i in range(tindex.ndims-1)] 
            branch_indices.extend(branch_layer) 

        return self.__getitem__(branch_indices)
    
    def get_layer(self, layer: int) -> dict[TTNIndex, torch.Tensor]:
        """
        Returns a dictionary of tensors and indices of the layer layer.
        """
        return {tindex: self.__tensor_map[tindex] for tindex in self.__indices if int(tindex.name.split('.')[0]) == layer}
    
    
    def _propagate_data_through_branch_(self, data: dict[TIndex, torch.Tensor], branch: dict[TTNIndex, torch.Tensor], keep=False, pbar=None, quantize=False) -> dict[TIndex, torch.Tensor] :
        """
        Propagates data through a branch of the TTN.
        """

        sorted_branch_keys = sorted(branch.keys(), reverse=True)
        last_idx = sorted_branch_keys[-1]
        branch_data = data | branch
        for tindex in sorted_branch_keys:
            if pbar is not None:
                pbar.set_postfix_str(f"contracting {tindex.name}")
            branch_data[tindex] = contract_up(branch_data[tindex].contiguous(), [branch_data[tindex[0]], branch_data[tindex[1]]], self.quantizer if quantize else None)
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(f"contracted {tindex.name}")
                

        if not keep:
            result = branch_data[last_idx].clone()
            del branch_data

        return {key: branch_data[key] for key in sorted_branch_keys} if keep else {TIndex(last_idx.name, last_idx[2]): result}

    
    def draw(self, name='TTN', cmap='viridis', fontsize=11):
        cmap = colormaps.get_cmap(cmap)
        categories = np.linspace(0.2, 1, self.__n_layers)
        dot = graphviz.Digraph(name, comment='TTN: ' + name, format='svg', engine='dot', renderer='cairo', graph_attr={'bgcolour': 'transparent', 'rankdir': 'LR', 'splines':'false', 'size':'16,14', 'ratio':'compress', 'fontname':'Arial'})
        dot.attr('node', shape='circle', width='0.35', fixedsize='true', fontsize=str(fontsize))
        dot.attr('edge', color='#bfbfbf', fontsize=str(fontsize-2))
        dot.edge('0.0', 'hide', label=self.label_tag)
        dot.node('hide', '', shape='plaintext')
        for tindex in self.__indices:
            c_rgba = list(cmap(categories[int(tindex.name.split('.')[0])]))

            dot.node(tindex.name, tindex.name, fillcolor=colors.rgb2hex(c_rgba), style='filled', color=colors.rgb2hex(adjust_lightness(c_rgba, amount=0.8)), penwidth='4')
            
            dot.edge(tindex[0], tindex.name, label=tindex[0]+f' [{self.__tensor_map[tindex].shape[0]}]', weight=str((int(tindex.name.split('.')[0])+1)**2))
            dot.edge(tindex[1], tindex.name, label=tindex[1]+f' [{self.__tensor_map[tindex].shape[1]}]', weight=str((int(tindex.name.split('.')[0])+1)**2))
        
        for i in range(2**self.__n_layers):
            dot.node(f'data.{i}', '', shape='plaintext', width='0.1', height='0.1')
        return dot
    
    def initialize(self, train_dl: torch.utils.data.DataLoader, loss_fn, epochs = 5, disable_pbar=False):
        # now we want to run across the ttn, layer by layer
        # and initialize the tensors by getting the partial dm
        # of two sites of the previous layer, diagonalizing it,
        # and isometrizing the rotation matrix (with n eigenvectors
        # corresponding to the n=bond_dim greatest eigenvalues)
        
        data = [data_batch.squeeze().to(self.device, dtype=self.__dtype) for data_batch, _ in train_dl]
        data_indices = [TIndex(f'data.{i}', [f'data.{i}']) for i in range(data[0].shape[1])]
        
        pbar = tqdm(
            total=(self.n_layers - 1) * len(train_dl)
                  + 2 * (2 ** (self.n_layers - 1) - 1),
            desc="ttn unsupervised init",
            position=0,
            leave=True,
            disable=disable_pbar,
        )
        for layer in range(self.n_layers - 1, 0, -1):  # do this for all layers except the uppermost one
            pbar.set_postfix_str(f"doing layer {layer}")
            next_layer_list = []
            ttn_curr_layer = self.get_layer(layer)
            # perform initialization of current layer with partial dm
            # of state at previous layer
            for tindex, tensor in ttn_curr_layer.items():
                pbar.set_postfix_str(f"doing layer {layer}, tensor {tindex.name.split('.')[1]}/{2**layer}")
                sel_sites = [int(index.split(".")[-1]) for index in tindex.indices[:2]]
                partial_dm = 0
                
                for data_batch in data:
                    partial_dm += sep_partial_dm_torch(sel_sites, data_batch, skip_norm=True, device=self.device).sum(dim=0)
                partial_dm /= np.prod([data_batch.shape[0] for data_batch in data], dtype=np.float64)
                # now we have to diagonalize the partial dm
                eigvecs = torch.linalg.eigh(partial_dm)[1].to(dtype=self.__dtype)
                del partial_dm
                # the eigenvectors matrix should be isometrized, but let's check it first
                if not torch.allclose(torch.eye(eigvecs.shape[0], device=self.device), torch.matmul(eigvecs , eigvecs.T.conj()).float()):
                    raise ValueError(f"eigenvectors matrix is not isometrized for tensor {tindex.name}")

                # now we have to select the n eigenvectors corresponding to the n greatest eigenvalues
                # and reshape, as the physical indices of the two sites are fused in the first index
                self.__tensor_map[tindex] = eigvecs[:, -tensor.shape[-1]:].reshape(tensor.shape)
                del eigvecs
                pbar.update(1)

            # calculate next propagation of data to this layer
            # with the updated tensors
            ttn_curr_layer = self.get_layer(layer)
            pbar.set_postfix_str(f"doing layer {layer}, propagating data")
            for data_batch in data:
                new_data_layer = self._propagate_data_through_branch_(dict(zip(data_indices, data_batch.unbind(1))), ttn_curr_layer, keep=True).values()
                next_layer_list.append(torch.stack(list(new_data_layer), 1))

                pbar.update(1)
            del data
            data = next_layer_list
            data_indices = [TIndex(tindex.name, tindex.indices[-1:]) for tindex in ttn_curr_layer.keys()]
        pbar.set_postfix_str(f'done unsupervised init!')
        pbar.close()
        
        # now we want to initialize the top tensor
        # ! not working properly
        pbar = tqdm(data, total=len(data), desc='ttn supervised init',position=0, disable=disable_pbar)
        top_tensor = self.__tensor_map['0.0']
        top_parameter = torch.nn.Parameter(top_tensor, requires_grad=True)
        optimizer = torch.optim.Adam([top_parameter], 1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=2, min_lr=1e-5, verbose=True)
        for epoch in range(epochs):
            pbar.set_postfix_str(f"doing epoch {epoch+1}/{epochs}")
            losses = one_epoch_one_tensor_torch(top_parameter, data, train_dl, optimizer, loss_fn, device=self.device, disable_pbar=disable_pbar)
            scheduler.step(np.array([loss.item() for loss in losses]).mean())
        self.__tensor_map['0.0'] = top_parameter.detach()
        
        self.__tensors = [self.__tensor_map[idx] for idx in self.__indices] # ? this is a bit of a hack, but it works
        self.__initialized = True


# class which represents a Tree Tensor Network
# as a torch.nn.Module, so that it can be used
# as a model in a torch training loop.
class TTNModel(torch.nn.Module, TTN):
    def __init__(
            self,
            n_features,
            n_phys=2,
            n_labels=2,
            label_tag="label",
            bond_dim=8,
            dtype=torch.cdouble,
            device="cpu",
            quantizer = None
    ):
        torch.nn.Module.__init__(self)
        TTN.__init__(self, n_features, n_phys, n_labels, label_tag, bond_dim, dtype, device, quantizer)
        self.model_init = True

    def initialize(self, dm_init=False, train_dl: torch.utils.data.Dataloader = None, loss_fn = None, epochs=5, disable_pbar=False):
        if dm_init:
            if (train_dl is None) or (loss_fn is None):
                raise ValueError(f"The unsupervised and supervised initialization were invoked but the dataloader was: {train_dl}\n and the loss function was: {loss_fn}")
            else:
                TTN.initialize(self, train_dl, loss_fn, epochs, disable_pbar=disable_pbar)
        
        super(TTNModel, type(self)).tensors.fset(self, torch.nn.ParameterList([torch.nn.Parameter(t, requires_grad=True) for t in self.tensors]))
        self.model_init = True

    def draw(self):
        return TTN.draw(self)

    def forward(self, x: torch.Tensor):
        if not self.model_init:
            raise RuntimeError("TTNModel not initialized")
        data_dict = {TIndex(f"data.{i}", [f"data.{i}"]): datum for i, datum in enumerate(x.unbind(1))}

        return self._propagate_data_through_branch_(data_dict, self.get_branch('0.0'), keep=True, quantize=True)['0.0']



########## FUNCTIONS FOR TTN ##########
#######################################
# function to check if the tensors of a TTN are correctly initialized
# by checking if the contractions of the tensors of each layer give
# the identity.
def check_correct_init(model: TTN):
    # gives true if correctly initialized and also the number of errors
    result_list = []
    for layer in range(model.n_layers-1,0,-1):
        layer_tensors = model.get_layer(layer)
        for tidx, tensor in enumerate(layer_tensors.values()):
            matrix = tensor.reshape(-1, tensor.shape[-1])
            contr = torch.matmul(matrix.T.conj(), matrix)
            result = torch.allclose(contr.data, torch.eye(contr.shape[-1], dtype=model.dtype, device='cuda'))
            if not result:
                print(f'Layer {layer}, tensor {tidx} is not initialized correctly')
            result_list.append(not result)
    
    n_errors = torch.tensor(result_list, dtype=torch.bool).sum().item()
    return n_errors == 0, n_errors