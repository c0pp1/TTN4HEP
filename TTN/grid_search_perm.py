import ttn_torch as ttn
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from utils import accuracy, train_one_epoch, get_higgs_data_loaders, class_loss
import time

from sklearn.metrics import roc_auc_score

DEVICE = 'cuda'
DEVICE = 'cuda' if torch.cuda.is_available() and DEVICE=='cuda' else 'cpu'
SCHEDULER_STEPS = 4
EPOCHS = 80
INIT_EPOCHS = 5
POPULATION = 5
LR = 0.02
DISABLE_PBAR = False
LAMBDA = 0.1
OUT_DIR = 'data/grid_search_perm/'

def get_predictions(model, device, dl, dtype=torch.complex128, disable_pbar=False):

    model.eval()
    model.to(device)
    predictions = []
    with torch.no_grad():
        for data in tqdm(dl, total=len(dl), position=0, desc='test', disable=disable_pbar):
            images, labels = data
            images, labels = images.to(device, dtype=dtype).squeeze(), labels.to(device)
            outputs = model(images)
            probs = torch.real(torch.pow(outputs, 2))
            #probs = probs / torch.sum(probs)
            predictions.append(probs.squeeze().detach().cpu())

    return torch.concat(predictions, dim=0)


def train(model: ttn.TTNModel, train_dl, pbar = None, disable_pbar=False):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=-1, verbose=False), torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=1e-5)]
    scheduler = schedulers[0]
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

    FEATURES = [8]
    imsize_dict = {4: (2,2), 8: (4, 2), 16: (4, 4), 32: (8, 4), 64: (8, 8)}
    BATCH_SIZES = [128]
    INITIALIZE = [True]
    DTYPES = [torch.double]
    BOND_DIMS_DICT = {4: [2, 3, 4], 8: [16], 16: [4, 8, 16, 64, 128, 256], 32: [4, 16, 64, 128, 256], 64: [4, 16, 64, 128, 256]}
    n_perm = 20
    PERMUTATIONS = [torch.randperm(8) for _ in range(n_perm)]

    df = pd.DataFrame(columns=['permutation', 'sample', 'loss', 'train_acc', 'test_acc', 'train_acc0', 'test_acc0', 'auc', 'time'])

    pbar = tqdm(total=np.sum([len(BOND_DIMS_DICT[feat]) for feat in FEATURES])*len(BATCH_SIZES)*len(INITIALIZE)*len(DTYPES)*POPULATION*n_perm, position=0, desc='grid searching', disable=DISABLE_PBAR)
    pbar_train = tqdm(total=EPOCHS, position=1, desc='training', disable=DISABLE_PBAR)

    for perm in PERMUTATIONS:
        for sample in range(POPULATION):

            pbar.set_postfix_str(f'perm: {perm.tolist()}, id: {sample}')

            train_dl, test_dl, features = get_higgs_data_loaders(batch_size=BATCH_SIZES[0], dtype=DTYPES[0], permutation=perm)
            pbar_train.reset()
            try:
                model = ttn.TTNModel(features, bond_dim=BOND_DIMS_DICT[features][0], n_labels=1, device=DEVICE, dtype=DTYPES[0])
                model.initialize(INITIALIZE[0], train_dl, class_loss, INIT_EPOCHS, disable_pbar=True)
                train_acc0, test_acc0 = accuracy(model, DEVICE, train_dl, test_dl, DTYPES[0], disable_pbar=True)
                model.train()
                model.to(DEVICE)
                
                start = time.time()
                loss_history, final_epoch_loss = train(model, train_dl, pbar = pbar_train, disable_pbar=True)
                end = time.time()
            except Exception as e:
                print(e)
                with open(OUT_DIR + 'failed.txt', 'a') as f:
                    f.write(f'permutation: {perm.tolist()}, sample: {sample}\n')
                    f.write(str(e))
                    f.write('\n')
                #df.to_csv(OUT_DIR + 'grid_search.csv')
                #df.to_pickle(OUT_DIR + 'grid_search.pkl')
                #exit(1)
                pbar.update(1)
                continue

            model.eval()
            train_acc, test_acc = accuracy(model, DEVICE, train_dl, test_dl, DTYPES[0], disable_pbar=True)
            y_true = torch.cat([y for _, y in test_dl], dim=0).numpy()
            y_pred = get_predictions(model, DEVICE, test_dl, DTYPES[0], disable_pbar=True).numpy()
            auc = roc_auc_score(y_true, y_pred)
            
            torch.save(model.state_dict(), OUT_DIR + f'model_p{perm.tolist()}_id{sample}.pth')
            np.save(OUT_DIR + f'loss_history_p{perm.tolist()}_id{sample}.npy', loss_history)

            df.loc[df.index.size] = [perm.tolist(), sample, final_epoch_loss, train_acc, test_acc, 
                                     train_acc0, test_acc0, auc, end-start]

            pbar.update(1)

    pbar_train.close()
    pbar.close()
    df.to_csv(OUT_DIR + 'grid_search.csv')
    df.to_pickle(OUT_DIR + 'grid_search.pkl')

if __name__ == '__main__':
    main()
