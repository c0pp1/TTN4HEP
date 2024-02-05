import ttn_torch as ttn
import torch
from qtorch.quant import fixed_point_quantize, Quantizer
from qtorch.optim import OptimLP
from qtorch import FixedPoint
import pandas as pd
from tqdm import tqdm
import numpy as np
from utils import accuracy, train_one_epoch, get_stripeimage_data_loaders
import time
import traceback

DEVICE = 'cuda'
DEVICE = 'cuda' if torch.cuda.is_available() and DEVICE=='cuda' else 'cpu'
SCHEDULER_STEPS = 4
EPOCHS = 80
INIT_EPOCHS = 5
POPULATION = 5
LR = 0.02
LAMBDA = 0.1
DISABLE_PBAR = False

def loss(labels, output: torch.Tensor, weights, l=0.1):
    loss_value = 0.
    # regularization
    if l > 0.:
        norm = 0
        for tensor in weights:
            norm += torch.norm(tensor)
        norm /= len(weights)
        loss_value += l*(norm-1.)**2

    # loss based on output dimension
    if len(output.squeeze().shape) > 1:
        loss_value += torch.mean(torch.sum((output.squeeze() - labels)**2), -1)/2 
    else:
        loss_value += torch.mean((output.squeeze() - labels)**2)/2
    
    return loss_value

def train(model: ttn.TTNModel, train_dl, pbar = None, disable_pbar=False):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=-1, verbose=False)
    tot_loss_history = []
    loss_history = 0
    for epoch in range(EPOCHS):
        loss_history = train_one_epoch(model, DEVICE, train_dl, lambda *x: loss(*x, l=LAMBDA), optimizer, pbar=None, disable_pbar=disable_pbar)
        tot_loss_history += loss_history
        if epoch % SCHEDULER_STEPS == SCHEDULER_STEPS-1:
            scheduler.step()
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix_str(f'loss: {np.array(loss_history).mean():.5f}')

    tot_loss_history = np.array(tot_loss_history)

    return tot_loss_history, np.array(loss_history).mean()

def train_Q(model: ttn.TTNModel, weight_quant, train_dl, pbar = None, disable_pbar=False):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # turn your optimizer into a low precision optimizer
    optimizer = OptimLP(optimizer,
                        weight_quant=weight_quant)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=-1, verbose=False)
    tot_loss_history = []
    loss_history = 0
    for epoch in range(EPOCHS):
        loss_history = train_one_epoch(model, DEVICE, train_dl, lambda *x: loss(*x, l=LAMBDA), optimizer, pbar=None, disable_pbar=disable_pbar)
        tot_loss_history += loss_history
        if epoch % SCHEDULER_STEPS == SCHEDULER_STEPS-1:
            scheduler.step()
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix_str(f'loss: {np.array(loss_history).mean():.5f}')

    tot_loss_history = np.array(tot_loss_history)

    return tot_loss_history, np.array(loss_history).mean()


def main():

    FEATURES = 8
    imsize = (4, 2)
    WLS = [4, 8, 16]
    BATCH_SIZES = [32, 256]
    QUANTIZE = [True, False]
    DTYPE = torch.float
    BOND_DIMS = [3, 4, 8, 16]

    df = pd.DataFrame(columns=['bond_dim', 'batch_size', 'quantize', 'wl', 'sample', 'loss', 'train_acc', 'test_acc', 'train_acc0', 'test_acc0', 'time'])

    pbar = tqdm(total=len(BOND_DIMS)*len(BATCH_SIZES)*(len(WLS)+1)*POPULATION, position=0, desc='grid searching', disable=DISABLE_PBAR)
    pbar_train = tqdm(total=EPOCHS, position=1, desc='training', disable=DISABLE_PBAR)
    for bond_dim in BOND_DIMS:
        for batch_size in BATCH_SIZES:
            for quant in QUANTIZE:
                for wl in WLS if quant else [None]:
                    for sample in range(POPULATION):

                        pbar.set_postfix_str(f'bd: {bond_dim}, bs: {batch_size}, quant: {quant}')

                        train_dl, test_dl, feat = get_stripeimage_data_loaders(*imsize, batch_size=batch_size, dtype=DTYPE)
                        pbar_train.reset()
                        try:
                            if quant:
                                
                                # define the quantization parameters
                                forward_num = FixedPoint(wl=wl, fl=wl-2)
                                backward_num = FixedPoint(wl=wl, fl=wl-2)

                                # Create a quantizer
                                Q = Quantizer(forward_number=forward_num, backward_number=backward_num,
                                              forward_rounding="nearest", backward_rounding="stochastic")
                                
                                weight_quant = lambda x : fixed_point_quantize(x, wl, wl-2, rounding="nearest")
                                
                                model = ttn.TTNModel(feat, bond_dim=bond_dim, n_labels=1, device=DEVICE, dtype=DTYPE, quantizer=Q)
                            else:
                                model = ttn.TTNModel(feat, bond_dim=bond_dim, n_labels=1, device=DEVICE, dtype=DTYPE)

                            model.initialize(True, train_dl, loss, INIT_EPOCHS, disable_pbar=True)
                            train_acc0, test_acc0 = accuracy(model, DEVICE, train_dl, test_dl, DTYPE, disable_pbar=True)
                            
                            model.to(DEVICE)
                            model.train()
                            
                            start = time.time()
                            if quant:
                                loss_history, final_epoch_loss = train_Q(model, weight_quant, train_dl, pbar = pbar_train, disable_pbar=True)
                            else:
                                loss_history, final_epoch_loss = train(model, train_dl, pbar = pbar_train, disable_pbar=True)
                            end = time.time()
                        except Exception as e:
                            print(e)
                            traceback.print_exc()
                            with open('data/grid_search_fp/failed.txt', 'a') as f:
                                f.write(f'feat: {feat}, bond_dim: {bond_dim}, batch_size: {batch_size}, quant: {quant}\n')
                                f.write(str(e))
                                f.write('\n')
                            #df.to_csv('data/grid_search/grid_search.csv')
                            #df.to_pickle('data/grid_search/grid_search.pkl')
                            #exit(1)
                            pbar.update(1)
                            continue

                        model.eval()
                        train_acc, test_acc = accuracy(model, DEVICE, train_dl, test_dl, DTYPE, disable_pbar=True)
                        
                        torch.save(model.state_dict(), f'data/grid_search_fp/model_bd{bond_dim}_bs{batch_size}_q{quant}_wl{wl}_id{sample}.pth')
                        np.save(f'data/grid_search_fp/loss_history_bd{bond_dim}_bs{batch_size}_q{quant}_wl{wl}_id{sample}.npy', loss_history)

                        df.loc[df.index.size] = [bond_dim, batch_size, quant, wl, sample, final_epoch_loss, 
                                                    train_acc, test_acc, train_acc0, test_acc0, end-start]

                        pbar.update(1)

    pbar_train.close()
    pbar.close()
    df.to_csv('data/grid_search_fp/grid_search.csv')
    df.to_pickle('data/grid_search_fp/grid_search.pkl')

if __name__ == '__main__':
    main()
