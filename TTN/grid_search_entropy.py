import ttn_torch as ttn
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from utils import accuracy, train_one_epoch, get_titanic_data_loaders, class_loss
import time
import traceback
from itertools import combinations_with_replacement
from sklearn.metrics import roc_auc_score
import os


DEVICE = 'cuda'
DEVICE = 'cuda' if torch.cuda.is_available() and DEVICE=='cuda' else 'cpu'
SCHEDULER_STEPS = 4
EPOCHS = 80
INIT_EPOCHS = 15
POPULATION = 5
LR = 0.02
LAMBDA = 0.1
DISABLE_PBAR = False
OUT_DIR = 'data/grid_search_entropy_titanic/'

def get_predictions(model, device, dl, dtype=torch.complex128, disable_pbar=False):

    model.eval()
    model.to(device)
    predictions = []
    with torch.no_grad():
        for data in tqdm(dl, total=len(dl), position=0, desc='test', disable=disable_pbar):
            images, labels = data
            images, labels = images.to(device, dtype=dtype).squeeze(), labels.to(device)
            outputs = model(images)
            probs = torch.pow(torch.abs(outputs), 2)
            #probs = probs / torch.sum(probs)
            predictions.append(probs.squeeze().detach().cpu())

    return torch.concat(predictions, dim=0)


def train(model: ttn.TTNModel, train_dl, pbar = None, disable_pbar=False):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=-1, verbose=False)
    tot_loss_history = []
    loss_history = 0
    for epoch in range(EPOCHS):
        loss_history = train_one_epoch(model, DEVICE, train_dl, lambda *x: class_loss(*x, l=LAMBDA), optimizer, pbar=None, disable_pbar=disable_pbar)
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
    BATCH_SIZES = [32]
    DTYPE = torch.double
    BOND_DIMS = [3, 4, 8, 16]
    PERMUTATIONS = [[0, 1, 2, 3, 4, 5, 6, 7],
                    [0, 2, 3, 4, 5, 6, 7, 1],
                    [0, 3, 4, 5, 6, 7, 1, 2],
                    [3, 1, 4, 2, 5, 6, 7, 0],
                    [4, 5, 0, 6, 7, 1, 2, 3],
                    [0, 4, 5, 6, 7, 1, 2, 3],
                    [7, 6, 2, 3, 4, 5, 0, 1],
                    [1, 7, 6, 2, 3, 4, 5, 0]]
    
    np.save(OUT_DIR + 'permutations.npy', np.array(PERMUTATIONS))

    sz = torch.tensor([[1, 0], [0, -1]], dtype = DTYPE, device = DEVICE)

    df = pd.DataFrame(columns=['bond_dim', 'perm', 'sample', 'loss', 'train_acc', 'test_acc', 'train_acc0', 'test_acc0', 'auc', 'time'])

    pbar = tqdm(total=len(BOND_DIMS)*len(BATCH_SIZES)*len(PERMUTATIONS)*POPULATION, position=0, desc='grid searching', disable=DISABLE_PBAR)
    pbar_train = tqdm(total=EPOCHS, position=1, desc='training', disable=DISABLE_PBAR)
    for bond_dim in BOND_DIMS:
        for batch_size in BATCH_SIZES:
            for i, perm in enumerate(PERMUTATIONS):
                for sample in range(POPULATION):

                    pbar.set_postfix_str(f'bd: {bond_dim}, bs: {batch_size}, perm: {i}, sample: {sample}')

                    train_dl, test_dl, feat = get_titanic_data_loaders(batch_size=batch_size, dtype=DTYPE, mapping='spin', permutation=perm)
                    pbar_train.reset()
                    try:
                        model = ttn.TTNModel(feat, bond_dim=bond_dim, n_labels=1, device=DEVICE, dtype=DTYPE)

                        model.initialize(True, train_dl, class_loss, INIT_EPOCHS, disable_pbar=True)
                        train_acc0, test_acc0 = accuracy(model, DEVICE, train_dl, test_dl, DTYPE, disable_pbar=True)
                        
                        model.to(DEVICE)
                        model.train()
                        
                        start = time.time()
                        
                        loss_history, final_epoch_loss = train(model, train_dl, pbar = pbar_train, disable_pbar=True)
                        end = time.time()
                    except Exception as e:
                        print(e)
                        traceback.print_exc()
                        with open(OUT_DIR + 'failed.txt', 'a') as f:
                            f.write(f'feat: {feat}, bond_dim: {bond_dim}, perm: {perm}\n')
                            f.write(str(e))
                            f.write('\n')
                        #df.to_csv('data/grid_search/grid_search.csv')
                        #df.to_pickle('data/grid_search/grid_search.pkl')
                        #exit(1)
                        pbar.update(1)
                        continue

                    model.eval()
                        
                    np.savez(OUT_DIR + f'entropy_bd{bond_dim}_perm{i}_id{sample}', **model.get_mi())

                    feat_combos = list(combinations_with_replacement(np.arange(model.n_features), 2))
                    corr = np.zeros((model.n_features, model.n_features))
                    for combo in feat_combos:
                        corr[combo] = model.expectation({ttn.TIndex(f'data.{i}', [f'data.{i}']): sz for i in (combo if combo[0] != combo[1] else (combo[0],))})

                    np.save(OUT_DIR + f'corr_bd{bond_dim}_perm{i}_id{sample}', corr)

                    train_acc, test_acc = accuracy(model, DEVICE, train_dl, test_dl, DTYPE, disable_pbar=True)
                    y_true = torch.cat([y for _, y in test_dl], dim=0).numpy()
                    y_pred = get_predictions(model, DEVICE, test_dl, DTYPE, disable_pbar=True).numpy()
                    auc = roc_auc_score(y_true, y_pred)
                    
                    torch.save(model.state_dict(), OUT_DIR + f'model_bd{bond_dim}_perm{i}_id{sample}.pth')
                    np.save(OUT_DIR + f'loss_history_bd{bond_dim}_perm{i}_id{sample}.npy', loss_history)

                    df.loc[df.index.size] = [bond_dim, i, sample, final_epoch_loss, train_acc, 
                                                test_acc, train_acc0, test_acc0, auc, end-start]

                    pbar.update(1)

    pbar_train.close()
    pbar.close()
    df.to_csv(OUT_DIR + 'grid_search.csv', mode='a', header=not os.path.exists(OUT_DIR + 'grid_search.csv'))
    df.to_pickle(OUT_DIR + 'grid_search.pkl')

if __name__ == '__main__':
    main()
