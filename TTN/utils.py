import torch
import torchvision as tv
from quimb import tensor as qtn
import numpy as np
from tqdm.autonotebook import tqdm
from functools import partial

class TTN(qtn.TensorNetwork):
    def __init__(self, n_features, n_phys=2, n_labels=2, label_tag='label', bond_dim=4, virtual=False, device='cpu'):

        if (n_features % 2) != 0:
            raise ValueError(f"n_features must be  power of 2, got: {n_features}")
        
        self.n_features = n_features
        self.n_phys     = n_phys
        self.n_layers   = int(np.log2(n_features))
        self.n_labels   = n_labels
        self.label_tag  = label_tag
        self.bond_dim   = bond_dim
        self.virtual    = virtual
        self.device     = device
        self.id         = np.zeros([self.bond_dim]*3)
        for i in range(self.id.shape[0]):
            self.id[i, i, i] = 1.


        # add first tensor with special index
        if not (self.n_layers-1):

            return qtn.TensorNetwork(
                [qtn.rand_tensor(shape = (self.n_phys, self.n_phys, self.n_labels),
                                inds  = ['00.000', '00.001', self.label_tag],
                                tags  = ['l0'],
                                dtype='complex128'
                )]
            )
        else:
            dim = min(self.n_phys**(self.n_layers), self.bond_dim)
            tensors = [qtn.rand_tensor(shape = (dim, dim, self.n_labels),
                                        inds  = ['00.000', '00.001', self.label_tag],
                                        tags  = ['l0'],
                                        dtype='complex128'
                        )]


        for l in range(1, self.n_layers-1): # constructing the ttn starting from the top
            
            dim_pre  = min(self.n_phys**(self.n_layers-l), self.bond_dim)
            dim_post = min(self.n_phys**(self.n_layers-l+1), self.bond_dim)
            tensors.extend([qtn.rand_tensor(
                                shape = [dim_pre]*2+[dim_post],
                                inds  = [f'{l:02}.{2*i:03}', f'{l:02}.{2*i+1:03}', f'{(l-1):02}.{i:03}'],
                                tags  = [f'l{l}'],
                                dtype='complex128'
                            ) if np.random.rand() < 0.5 else \
                            qtn.Tensor(
                                data  = self.id[:dim_pre, :dim_pre, :dim_post],
                                inds  = [f'{l:02}.{2*i:03}', f'{l:02}.{2*i+1:03}', f'{(l-1):02}.{i:03}'],
                                tags  = [f'l{l}']
                            ) for i in range(2**l)]
            )

        dim = min(self.n_phys**2, self.bond_dim)
        tensors.extend([qtn.rand_tensor(shape = [self.n_phys]*2+[dim],
                                inds  = [f'{self.n_layers-1:02}.{2*i:03}', f'{self.n_layers-1:02}.{2*i+1:03}', f'{(self.n_layers-2):02}.{i:03}'],
                                tags  = [f'l{self.n_layers-1}'],
                                dtype='complex128'
                        ) for i in range(2**(self.n_layers-1))]
                        )

        super(TTN, self).__init__(tensors, virtual=virtual)
        self.apply_to_arrays(lambda x: torch.tensor(x, device=device, dtype=torch.complex128))

    def draw(self, return_fig=False, ax=None):
        fig = super().draw(color=self.tags, show_inds='all', 
            fix=dict({ key : (4*i/self.n_features-1., -1.) for i, key in enumerate(list(self.tensor_map.keys())[:-self.n_features//2-1:-1])}, **{'l0':(0., 1.)}),
            return_fig=return_fig,
            ax=ax,
            dim=2,
            figsize=(15,10)
        )
        return fig
    
    def copy(self):
        return TTN(self.n_features, self.n_phys, self.n_labels, self.label_tag, self.bond_dim, self.virtual, self.device)
    



class TNModel(torch.nn.Module):

    def __init__(self, myttn: TTN):
        super().__init__()
        # extract the raw arrays and a skeleton of the TN
        self.myttn = myttn
        
        params, self.skeleton = qtn.pack(self.myttn)
        # n.b. you might want to do extra processing here to e.g. store each
        # parameter as a reshaped matrix (from left_inds -> right_inds), for 
        # some optimizers, and for some torch parametrizations
        self.torch_params = torch.nn.ParameterDict({
            # torch requires strings as keys
            str(i): torch.nn.Parameter(initial)
            for i, initial in params.items()
        })

    def old_forward(self, x: torch.Tensor):
        # convert back to original int key format
        params = {int(i): p for i, p in self.torch_params.items()}
        # reconstruct the TN with the new parameters
        tn = qtn.unpack(params, self.skeleton)
        results=[]

        for datum in x:
            # adapt datum to mps
            for _ in range(4 - datum.dim()):
                datum = datum.unsqueeze(0)
            contr = (qtn.MatrixProductState(torch.unbind(datum, -2), site_ind_id=f'{self.myttn.n_layers-1:02}.{{:03}}') & tn) ^ ...
            results.append(contr.data)
        
        return torch.stack(results)
    
    # cleaner forward function but no perfromance improvement, uff...
    def forward(self, x: torch.Tensor):
        # convert back to original int key format
        params = {int(i): p for i, p in self.torch_params.items()}
        # reconstruct the TN with the new parameters
        tn = qtn.unpack(params, self.skeleton)

        # construct data tn
        data_tn = qtn.TensorNetwork([qtn.Tensor(data=site_batch, inds=['b', f'{self.myttn.n_layers-1:02}.{i:03}'], tags='data') 
                              for i, site_batch in enumerate(torch.unbind(x, -2))])
        
        # contract data tn with model tn
        result = (tn & data_tn).contract(tags=..., output_inds=['b', self.myttn.label_tag])

        return result.data

    def draw(self, return_fig=False):
        fig = self.myttn.draw(return_fig=return_fig)
        return fig
    

############# DATASET HANDLING #############
############################################

def linearize(tensor: torch.Tensor):

    result = torch.clone(tensor).reshape((-1, np.prod(tensor.shape[-2:])))
    index = torch.as_tensor(range(result.shape[-1]))
    mask = index // 2 % 2 == 0

    for i in range(tensor.shape[-1]):
        result[:, (mask != i%2) & (index < (i//2+1)*2*tensor.shape[-1]) & (index >= (i//2)*2*tensor.shape[-1])]= tensor[:,i,:]

    return result

def quantize(tensor):

    cos = torch.cos(torch.pi*tensor/2)
    sin = torch.sin(torch.pi*tensor/2)

    return torch.stack([cos, sin], dim=-1)

def load_to_device(tensor: torch.Tensor, device):
    return tensor.to(device)

def balance(labels, train, test):
    # this function balances the specified labels in the train and test sets
    # it assumes that the labels are integers
    # labels: list of labels to balance
    # train: training set
    # test: test set
    # returns: balanced train and test sets

    # get the number of samples in each class
    train_class_counts = np.bincount(train.targets)
    test_class_counts = np.bincount(test.targets)

    # get the maximum number of samples in a class
    train_max_class_count = min(train_class_counts[labels])
    test_max_class_count = min(test_class_counts[labels])

    # get the indices of the samples in each class
    train_indices = [np.where(np.array(train.targets) == label)[0] for label in labels]
    test_indices = [np.where(np.array(test.targets) == label)[0] for label in labels]
    
    # get the indices of the samples in each class 
    # that will be used for training
    train_indices_balanced = [np.random.choice(train_index, train_max_class_count) for train_index in train_indices]
    test_indices_balanced = [np.random.choice(test_index, test_max_class_count) for test_index in test_indices]

    # get the balanced training and test sets
    train_balanced = torch.utils.data.Subset(train, np.concatenate(train_indices_balanced))
    test_balanced = torch.utils.data.Subset(test, np.concatenate(test_indices_balanced))

    return train_balanced, test_balanced


def get_ttn_transform(h, device='cpu'):
    return tv.transforms.Compose([tv.transforms.Resize((h, h)), 
                                   tv.transforms.ToTensor(),
                                   tv.transforms.Lambda(partial(load_to_device, device=device)),
                                   tv.transforms.Lambda(linearize), 
                                   tv.transforms.Lambda(quantize)]
            )
def get_ttn_transform_visual(h):
    return tv.transforms.Compose([tv.transforms.Resize((h, h)), 
                                   tv.transforms.ToTensor()]
            )


# kronecker product for leading batch dimension
def kron(a: torch.Tensor, b: torch.Tensor, batchs: int):

    ndb, nda = b.ndim, a.ndim
    nd = max(ndb, nda)-1 # suppose the first dimension is batch dimension

    if (nda == 0 or ndb == 0):
        return torch.multiply(a, b)

    as_ = a.shape
    bs = b.shape

    # Equalise the shapes by prepending smaller one with 1s
    as_ = (as_[0],) + (1,)*max(0, ndb-nda) + as_[1:]
    bs = (bs[0],) + (1,)*max(0, nda-ndb) + bs[1:]

    # Insert empty dimensions
    a_arr = a.view(as_) 
    b_arr = b.view(bs)
    
    # Compute the product
    for axis in range(1, nd*2, 2):
        a_arr = a_arr.unsqueeze(1+axis)
    for axis in range(0, nd*2, 2):
        b_arr = b_arr.unsqueeze(1+axis)
    
    result = torch.multiply(a_arr, b_arr)

    # Reshape back
    result = result.reshape((batchs,) + tuple(np.multiply(as_[1:], bs[1:])))

    return result



def sep_partial_dm(keep_index, sep_states: torch.utils.data.DataLoader | torch.Tensor | qtn.TensorNetwork, skip_norm=False, device='cpu'):
    if not isinstance(keep_index, torch.Tensor):
        keep_index = torch.tensor(keep_index, device=device, dtype=torch.int64)
    if isinstance(sep_states, torch.utils.data.DataLoader):
        discard_index = torch.ones(next(iter(sep_states))[0].shape[-2], dtype=torch.bool)
        discard_index[keep_index] = False
        rho_list = []
        for batch in tqdm(sep_states, desc='sep_partial_dm', position=1):
            batch = batch[0].to(device)
            if skip_norm:
                norm_factor = torch.eye(1, device=device)
            else:
                norm_factor = torch.prod(torch.sum(batch[...,discard_index,:]**2, dim=-1), dim=-1).squeeze()

            rhos = torch.einsum('...i,...j->...ij', batch[...,keep_index,:].conj(), batch[...,keep_index,:])
            rho = torch.eye(1, device=device, dtype=torch.complex128)
            
            for i in keep_index-keep_index.min():   # strange way to index but in this way we can get the partial density matrix also for different permutations of the sites
                rho = kron(rho, rhos[...,i,:,:], batchs=batch.shape[0])
            
            rho_list.append(rho*norm_factor.view([-1]+[1]*(rho.ndim-norm_factor.ndim)))
        return torch.concat(rho_list, dim=0)
    elif isinstance(sep_states, qtn.TensorNetwork):
        batch = [tensor.data for tensor in sep_states]
        batch = torch.stack(batch, dim=-2) # suppose the sites dimension is the second to last
        return sep_partial_dm(keep_index, batch, skip_norm=skip_norm, device=device)
    elif isinstance(sep_states, torch.Tensor):
        batch = sep_states.to(device)
        if skip_norm:
            norm_factor = torch.eye(1, device=device, dtype=torch.complex128)
        else:
            discard_index = torch.ones(sep_states.shape[-2], dtype=torch.bool)
            discard_index[keep_index] = False
            norm_factor = torch.prod(torch.sum(batch[...,discard_index,:]**2, dim=-1), dim=-1).squeeze()
        
        rhos = torch.einsum('...i,...j->...ij', batch[...,keep_index,:].conj(), batch[...,keep_index,:])
        rho = torch.eye(1, device=device)
        for i in keep_index-keep_index.min():
            rho = kron(rho, rhos[...,i,:,:], batchs=batch.shape[0])

        return rho*norm_factor.view([-1]+[1]*(rho.ndim-norm_factor.ndim))
    else:
        raise TypeError(f"sep_states must be one of torch.utils.data.DataLoader, torch.Tensor or quimb.tensor.TensorNetwork, got: {type(sep_states)}")