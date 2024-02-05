import ttn_torch as ttn
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from utils import accuracy, train_one_epoch, get_stripeimage_data_loaders
import time

DEVICE = 'cuda'
DEVICE = 'cuda' if torch.cuda.is_available() and DEVICE=='cuda' else 'cpu'
SCHEDULER_STEPS = 4
EPOCHS = 80
INIT_EPOCHS = 5
POPULATION = 5
LR = 0.02
DISABLE_PBAR = False
LAMBDA = 0.1
OUT_DIR = 'data/grid_search2/'

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
    schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=-1, verbose=False), torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=1e-5)]
    scheduler = schedulers[0]
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

    FEATURES = [4, 8, 16, 32]
    imsize_dict = {4: (2,2), 8: (4, 2), 16: (4, 4), 32: (8, 4), 64: (8, 8)}
    BATCH_SIZES = [16, 128]
    INITIALIZE = [True]
    DTYPES = [torch.double]
    BOND_DIMS_DICT = {4: [2, 3, 4], 8: [3, 4, 8, 16], 16: [4, 8, 16, 64, 128, 256], 32: [4, 16, 64, 128, 256], 64: [4, 16, 64, 128, 256]}

    df = pd.DataFrame(columns=['features', 'bond_dim', 'batch_size', 'initialize', 'dtype', 'sample', 'loss', 'train_acc', 'test_acc', 'train_acc0', 'test_acc0', 'time'])

    pbar = tqdm(total=np.sum([len(BOND_DIMS_DICT[feat]) for feat in FEATURES])*len(BATCH_SIZES)*len(INITIALIZE)*len(DTYPES)*POPULATION, position=0, desc='grid searching', disable=DISABLE_PBAR)
    pbar_train = tqdm(total=EPOCHS, position=1, desc='training', disable=DISABLE_PBAR)
    for feat in FEATURES:
        for bond_dim in BOND_DIMS_DICT[feat]:
            for dtype in DTYPES:
                for batch_size in BATCH_SIZES:
                    for init in INITIALIZE:
                        for sample in range(POPULATION):

                            pbar.set_postfix_str(f'f: {feat}, bd: {bond_dim}, t: {dtype}, bs: {batch_size}, init: {init}, id: {sample}')

                            train_dl, test_dl, features = get_stripeimage_data_loaders(*imsize_dict[feat], batch_size=batch_size, dtype=dtype)
                            pbar_train.reset()
                            try:
                                model = ttn.TTNModel(feat, bond_dim=bond_dim, n_labels=1, device=DEVICE, dtype=dtype)
                                model.initialize(init, train_dl, loss, INIT_EPOCHS, disable_pbar=True)
                                train_acc0, test_acc0 = accuracy(model, DEVICE, train_dl, test_dl, dtype, disable_pbar=True)
                                model.train()
                                model.to(DEVICE)
                                
                                start = time.time()
                                loss_history, final_epoch_loss = train(model, train_dl, pbar = pbar_train, disable_pbar=True)
                                end = time.time()
                            except Exception as e:
                                print(e)
                                with open(OUT_DIR + 'failed.txt', 'a') as f:
                                    f.write(f'feat: {feat}, bond_dim: {bond_dim}, dtype: {dtype}, batch_size: {batch_size}, init: {init}, sample: {sample}\n')
                                    f.write(str(e))
                                    f.write('\n')
                                #df.to_csv(OUT_DIR + 'grid_search.csv')
                                #df.to_pickle(OUT_DIR + 'grid_search.pkl')
                                #exit(1)
                                pbar.update(1)
                                continue

                            model.eval()
                            train_acc, test_acc = accuracy(model, DEVICE, train_dl, test_dl, dtype, disable_pbar=True)
                            
                            torch.save(model.state_dict(), OUT_DIR + f'model_f{feat}_bd{bond_dim}_bs{batch_size}_i{init}_t{dtype}_id{sample}.pth')
                            np.save(OUT_DIR + f'loss_history_f{feat}_bd{bond_dim}_bs{batch_size}_i{init}_t{dtype}_id{sample}.npy', loss_history)

                            df.loc[df.index.size] = [feat, bond_dim, batch_size, init, dtype, sample, final_epoch_loss, 
                                                        train_acc, test_acc, train_acc0, test_acc0, end-start]

                            pbar.update(1)

    pbar_train.close()
    pbar.close()
    df.to_csv(OUT_DIR + 'grid_search.csv')
    df.to_pickle(OUT_DIR + 'grid_search.pkl')

if __name__ == '__main__':
    main()
