import os
import ttn_torch as ttn
import torch
import pandas as pd
from tqdm import tqdm, trange
import numpy as np
from utils import accuracy, train_one_epoch, get_stripeimage_data_loaders, class_loss_fn, dclass_loss_fn
import time

DEVICE = 'cuda'
DEVICE = 'cuda' if torch.cuda.is_available() and DEVICE=='cuda' else 'cpu'
SCHEDULER_STEPS = 5
EPOCHS = 50
SWEEPS = 10
INIT_EPOCHS = 10
LR = 0.05
DISABLE_PBAR = False
LAMBDA = 0.01
OUT_DIR = 'data/grid_search3/'


def train(model: ttn.TTNModel, train_dl, pbar = None, disable_pbar=False):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=-1, verbose=False)
    tot_loss_history = []
    loss_history = 0
    for epoch in range(EPOCHS):
        loss_history = train_one_epoch(model, DEVICE, train_dl, lambda *x: class_loss_fn(*x, l=LAMBDA), optimizer, pbar=None, disable_pbar=disable_pbar)
        tot_loss_history += loss_history
        if epoch % SCHEDULER_STEPS == SCHEDULER_STEPS-1:
            scheduler.step()
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix_str(f'loss: {np.array(loss_history).mean():.5f}')

    tot_loss_history = np.array(tot_loss_history)

    return tot_loss_history, np.array(loss_history).mean()


def main():

    FEATURES = [4, 8, 16, 32, 64, 128]
    imsize_dict = {4: (2,2), 8: (4, 2), 16: (4, 4), 32: (8, 4), 64: (8, 8), 128: (8, 16)}
    BATCH_SIZES = [64, 1024]
    INITIALIZE = [True, False]
    DTYPES = [torch.double, torch.float32]
    BOND_DIMS_DICT = {4: [2, 3, 4], 8: [3, 4, 8, 16], 16: [4, 8, 16, 64, 128, 256], 32: [4, 16, 64, 128, 256], 64: [4, 16, 256], 128: [16, 256]}
    SWEEP = [True, False]
    MAPPINGS = ['spin', 'poly']

    df = pd.DataFrame(columns=['features', 'bond_dim', 'batch_size', 'initialize', 'dtype', 'sweep', 'map', 'loss', 'train_acc', 'test_acc', 'train_acc0', 'test_acc0', 'time'])

    pbar = tqdm(total=np.sum([len(BOND_DIMS_DICT[feat]) for feat in FEATURES])*len(BATCH_SIZES)*len(INITIALIZE)*len(DTYPES)*len(SWEEP)*len(MAPPINGS), position=0, desc='grid searching', disable=DISABLE_PBAR)
    
    for feat in FEATURES:
        for bond_dim in BOND_DIMS_DICT[feat]:
            for dtype in DTYPES:
                for batch_size in BATCH_SIZES:
                    for init in INITIALIZE:
                        for sweep in SWEEP:
                            for mapping in MAPPINGS:

                                pbar.set_postfix_str(f'f: {feat}, bd: {bond_dim}, t: {dtype}, bs: {batch_size}, init: {init}, sw: {sweep}, map: {mapping}')

                                train_dl, test_dl, features = get_stripeimage_data_loaders(*imsize_dict[feat], batch_size=batch_size, dtype=dtype, mapping=mapping)

                                try:
                                    model = ttn.TTNModel(feat, bond_dim=bond_dim, n_labels=1, device=DEVICE, dtype=dtype)
                                    model.initialize(init, train_dl, lambda *x: class_loss_fn(*x, l=LAMBDA), INIT_EPOCHS, disable_pbar=True)
                                    train_acc0, test_acc0 = accuracy(model, DEVICE, train_dl, test_dl, dtype, disable_pbar=True)
                                    model.train()
                                    model.to(DEVICE)
                                    
                                    start = time.time()
                                    if sweep:
                                        tot_loss_history = []
                                        optimizer = torch.optim.Adam(model.parameters(), lr=2**(model.n_layers-6))
                                        pbar_train_sweep = tqdm(total=SWEEPS, position=1, desc='sweeping', disable=DISABLE_PBAR)
                                        for s in range(SWEEPS):
                                            loss_history, _ = model.sweep(train_dl, dclass_loss_fn, optimizer, epochs=2, path_type='layer', manual=True, loss=lambda*x: class_loss_fn(*x, l=0.), verbose=0, save_grads=False)
                                            tot_loss_history += loss_history
                                            pbar_train_sweep.update(1)
                                            pbar_train_sweep.set_postfix_str(f'loss: {np.array(loss_history).mean():.4f}')
                                        pbar_train_sweep.close()
                                        final_epoch_loss = np.array(loss_history).mean()
                                    else:
                                        pbar_train = tqdm(total=EPOCHS, position=1, desc='training', disable=DISABLE_PBAR)
                                        loss_history, final_epoch_loss = train(model, train_dl, pbar = pbar_train, disable_pbar=True)
                                        pbar_train.close()
                                    end = time.time()
                                except Exception as e:
                                    print(e)
                                    with open(OUT_DIR + 'failed.txt', 'a') as f:
                                        f.write(f'feat: {feat}, bond_dim: {bond_dim}, dtype: {dtype}, batch_size: {batch_size}, init: {init}, sweep: {sweep}, map: {mapping}\n')
                                        f.write(str(e))
                                        f.write('\n')
                                    #df.to_csv(OUT_DIR + 'grid_search.csv')
                                    #df.to_pickle(OUT_DIR + 'grid_search.pkl')
                                    #exit(1)
                                    pbar.update(1)
                                    continue

                                model.eval()
                                train_acc, test_acc = accuracy(model, DEVICE, train_dl, test_dl, dtype, disable_pbar=True)
                                
                                model.to_npz(OUT_DIR + f'model_f{feat}_bd{bond_dim}_bs{batch_size}_i{init}_t{dtype}_sw{sweep}_map{mapping}.npz')
                                #torch.save(model.state_dict(), OUT_DIR + f'model_f{feat}_bd{bond_dim}_bs{batch_size}_i{init}_t{dtype}_sw{sweep}_map{mapping}.pth')
                                np.save(OUT_DIR + f'loss_history_f{feat}_bd{bond_dim}_bs{batch_size}_i{init}_t{dtype}_sw{sweep}_map{mapping}.npy', loss_history)

                                df.loc[df.index.size] = [feat, bond_dim, batch_size, init, dtype, sweep, mapping, final_epoch_loss, 
                                                         train_acc, test_acc, train_acc0, test_acc0, end-start]

                                pbar.update(1)

    pbar.close()
    df.to_csv(OUT_DIR + 'grid_search.csv')
    df.to_pickle(OUT_DIR + 'grid_search.pkl')

if __name__ == '__main__':
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    main()
