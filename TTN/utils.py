import torch
from quimb import tensor as qtn
import numpy as np

class TTN(qtn.TensorNetwork):
    def __init__(self, n_features, n_phys=2, n_labels=2, bond_dim=4):

        super(TTN, self).__init__()

        if (n_features % 2) != 0:
            raise ValueError(f"n_features must be  power of 2, got: {n_features}")
        
        self.n_features = n_features
        self.n_phys     = n_phys
        self.n_hlayers  = int(np.log2(n_features))-1
        self.n_labels   = n_labels
        self.bond_dim   = bond_dim
        self.id         = np.zeros([self.bond_dim]*3, dtype='complex128')
        for i in range(self.id.shape[0]):
            self.id[i, i, i] = 1.


    def build(self):

        # add first tensor with special index
        if not self.n_hlayers:

            return qtn.TensorNetwork(
                [qtn.rand_phased(shape = (self.n_phys, self.n_phys, self.n_labels),
                                inds  = ['00.000', '00.001', 'label'],
                                tags  = ['l0']
                )]
            )
        else:
            tensors = [qtn.rand_phased(shape = (self.bond_dim, self.bond_dim, self.n_labels),
                                       inds  = ['00.000', '00.001', 'label'],
                                       tags  = ['l0']
                       )]


        for l in range(1, self.n_hlayers):
            
            tensors.extend([qtn.rand_phased(
                                shape = [self.bond_dim]*3,
                                inds  = [f'{l:02}.{2*i:03}', f'{l:02}.{2*i+1:03}', f'{(l-1):02}.{i:03}'],
                                tags  = [f'l{l}']
                            ) if np.random.rand() < 0.5 else \
                            qtn.Tensor(
                                data = self.id,
                                inds  = [f'{l:02}.{2*i:03}', f'{l:02}.{2*i+1:03}', f'{(l-1):02}.{i:03}'],
                                tags  = [f'l{l}']
                            ) for i in range(2**l)]
            )


        tensors.extend([qtn.rand_phased(shape = [self.n_phys]*2+[self.bond_dim],
                                inds  = [f'{self.n_hlayers:02}.{2*i:03}', f'{self.n_hlayers:02}.{2*i+1:03}', f'{(self.n_hlayers-1):02}.{i:03}'],
                                tags  = [f'l{self.n_hlayers}']
                        ) for i in range(2**self.n_hlayers)]
                        )

        self.net = qtn.TensorNetwork(tensors)
        return self.net

        


class TNModel(torch.nn.Module):

    def __init__(self, myttn: TTN):
        super().__init__()
        # extract the raw arrays and a skeleton of the TN
        self.myttn = myttn
        net = myttn.build()
        net.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
        
        params, self.skeleton = qtn.pack(net)
        # n.b. you might want to do extra processing here to e.g. store each
        # parameter as a reshaped matrix (from left_inds -> right_inds), for 
        # some optimizers, and for some torch parametrizations
        self.torch_params = torch.nn.ParameterDict({
            # torch requires strings as keys
            str(i): torch.nn.Parameter(initial)
            for i, initial in params.items()
        })

    def forward(self, x: torch.Tensor):
        # convert back to original int key format
        params = {int(i): p for i, p in self.torch_params.items()}
        # reconstruct the TN with the new parameters
        tn = qtn.unpack(params, self.skeleton)
        results=[]

        for datum in x:
            # adapt datum to mps
            for _ in range(4 - datum.dim()):
                datum = datum.unsqueeze(0)
            contr = (qtn.MatrixProductState(torch.unbind(datum, -2), site_ind_id=f'{self.myttn.n_hlayers:02}.{{:03}}') & tn) ^ ...
            results.append(contr.data)
        
        return torch.stack(results)

    def draw(self, return_fig=False):
        net = self.myttn.net
        fig = net.draw(color=net.tags, show_inds='all', 
            fix=dict({ key : (4*i/self.myttn.n_features-1., -1.) for i, key in enumerate(list(net.tensor_map.keys())[:-self.myttn.n_features//2-1:-1])}, **{'l0':(0., 1.)}),
            return_fig=return_fig,
            dim=2,
            figsize=(15,10)
        )
        return fig