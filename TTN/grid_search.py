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
#POPULATION = 5
LR = 0.02
DISABLE_PBAR = False

def loss(labels, output):
    return torch.mean((output.squeeze() - labels)**2)/2

def train(model: ttn.TTNModel, train_dl, pbar = None, disable_pbar=False):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=-1, verbose=False), torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=1e-5)]
    scheduler = schedulers[0]
    tot_loss_history = []
    loss_history = 0
    for epoch in range(EPOCHS):
        loss_history = train_one_epoch(model, DEVICE, train_dl, loss, optimizer, pbar=None, disable_pbar=disable_pbar)
        tot_loss_history += loss_history
        if epoch % SCHEDULER_STEPS == SCHEDULER_STEPS-1:
            scheduler.step()
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix_str(f'loss: {np.array(loss_history).mean():.5f}')

    tot_loss_history = np.array(tot_loss_history)

    return tot_loss_history, np.array(loss_history).mean()


def main():

    FEATURES = [4, 8, 16, 32, 64]
    imsize_dict = {4: (2,2), 8: (4, 2), 16: (4, 4), 32: (8, 4), 64: (8, 8)}
    BATCH_SIZES = [16, 32, 128, 256]
    INITIALIZE = [True, False]
    DTYPES = [torch.double, torch.cdouble]
    BOND_DIMS_DICT = {4: [2, 3, 4], 8: [3, 4, 8, 16], 16: [4, 8, 16, 64, 128, 256], 32: [4, 16, 64, 128, 256], 64: [4, 16, 64, 128, 256]}

    df = pd.DataFrame(columns=['features', 'bond_dim', 'batch_size', 'initialize', 'dtype', 'loss', 'train_acc', 'test_acc', 'train_acc0', 'test_acc0', 'time'])

    pbar = tqdm(total=np.sum([len(BOND_DIMS_DICT[feat]) for feat in FEATURES])*len(BATCH_SIZES)*len(INITIALIZE)*len(DTYPES), position=0, desc='grid searching', disable=DISABLE_PBAR)
    pbar_train = tqdm(total=EPOCHS, position=1, desc='training', disable=DISABLE_PBAR)
    for feat in FEATURES:
        for bond_dim in BOND_DIMS_DICT[feat]:
            for dtype in DTYPES:
                for batch_size in BATCH_SIZES:
                    for init in INITIALIZE:

                        pbar.set_postfix_str(f'feat: {feat}, bond_dim: {bond_dim}, dtype: {dtype}, batch_size: {batch_size}, init: {init}')

                        train_dl, test_dl = get_stripeimage_data_loaders(*imsize_dict[feat], batch_size=batch_size, dtype=dtype)
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
                            with open('data/grid_search/failed.txt', 'a') as f:
                                f.write(f'feat: {feat}, bond_dim: {bond_dim}, dtype: {dtype}, batch_size: {batch_size}, init: {init}\n')
                                f.write(str(e))
                                f.write('\n')
                            #df.to_csv('data/grid_search/grid_search.csv')
                            #df.to_pickle('data/grid_search/grid_search.pkl')
                            #exit(1)
                            pbar.update(1)
                            continue

                        model.eval()
                        train_acc, test_acc = accuracy(model, DEVICE, train_dl, test_dl, dtype, disable_pbar=True)
                        
                        torch.save(model.state_dict(), f'data/grid_search/model_{feat}_{bond_dim}_{batch_size}_{init}_{dtype}.pth')
                        np.save(f'data/grid_search/loss_history_{feat}_{bond_dim}_{batch_size}_{init}_{dtype}.npy', loss_history)

                        df.loc[df.index.size] = [feat, bond_dim, batch_size, init, dtype, final_epoch_loss, 
                                                    train_acc, test_acc, train_acc0, test_acc0, end-start]

                        pbar.update(1)

    pbar_train.close()
    pbar.close()
    df.to_csv('data/grid_search/grid_search.csv')
    df.to_pickle('data/grid_search/grid_search.pkl')

if __name__ == '__main__':
    main()
