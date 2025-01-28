import ttn_torch as ttn
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from utils import accuracy, train_one_epoch, get_titanic_data_loaders, class_loss_fn, dclass_loss_fn, titanic_features
import time
import traceback
from itertools import combinations_with_replacement
from sklearn.metrics import roc_auc_score
import os


BATCH_SIZE = 100
DATASET = 'titanic'
MAPPING = 'spin'
MAP_DIM = 2
BOND_DIM = 8
L = 0.01
DEVICE = 'cuda'
DEVICE = 'cuda' if torch.cuda.is_available() and DEVICE=='cuda' else 'cpu'
SCHEDULER_STEPS = 4
EPOCHS = 2
SWEEPS = 30
INIT_EPOCHS = 15
DISABLE_PBAR = False
OUT_DIR = 'data/search_entropy/'
if OUT_DIR[-1] != '/':
    OUT_DIR += '/'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)


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

    loss = lambda*x: class_loss_fn(*x, l=0.)
    optimizer = torch.optim.Adam(model.parameters(), 2**(model.n_layers-6))
    loss_history = []
    if pbar is None:
        pbar = tqdm(total=SWEEPS, position=1, desc='sweeps', disable=disable_pbar)
    for sweep in range(SWEEPS):
        losses, _ = model.sweep(train_dl, dclass_loss_fn, optimizer, epochs=EPOCHS, path_type='layer+0', manual=True, loss=loss, verbose=0 if disable_pbar else 1, save_grads=False)
        loss_history.extend(losses)

        pbar.update(1)
        pbar.set_postfix_str(f'loss: {np.array(loss_history).mean():.5f}')

    tot_loss_history = np.array(loss_history)

    return tot_loss_history, np.array(loss_history).mean()


def main():

    OFFSET = 100
    SHOTS=100
    DTYPE = torch.double

    df = pd.DataFrame(columns=['sample', 'loss', 'train_acc', 'test_acc', 'auc'] + [f'entropy_{feat}' for feat in titanic_features])

    loss = lambda *x: class_loss_fn(*x, l=L)

    pbar = tqdm(total=SHOTS, position=0, desc='grid searching', disable=DISABLE_PBAR)
    pbar_train = tqdm(total=SWEEPS, position=1, desc='training', disable=DISABLE_PBAR)
    best_agesex_ent = 0.
    best_age_ent = 0.
    best_sex_ent = 0.
    best_id = -1

    for sample in range(OFFSET, OFFSET+SHOTS):
        train_dl, test_dl, feat = get_titanic_data_loaders(batch_size=BATCH_SIZE, dtype=DTYPE, mapping=MAPPING)
        pbar_train.reset()
        try:
            model = ttn.TTNModel(feat, bond_dim=BOND_DIM, n_labels=1, device=DEVICE, dtype=DTYPE)

            model.initialize(True, train_dl, loss, INIT_EPOCHS, disable_pbar=True)
            model.train()
            
            loss_history, final_epoch_loss = train(model, train_dl, pbar = pbar_train, disable_pbar=True)

        except Exception as e:
            print(e)
            traceback.print_exc()
            #df.to_csv('data/grid_search/grid_search.csv')
            #df.to_pickle('data/grid_search/grid_search.pkl')
            #exit(1)
            pbar.update(1)
            continue

        model.eval()

        train_acc, test_acc = accuracy(model, DEVICE, train_dl, test_dl, DTYPE, disable_pbar=True)

        y_true = torch.cat([y for _, y in test_dl], dim=0).numpy()
        y_pred = get_predictions(model, DEVICE, test_dl, DTYPE, disable_pbar=True).numpy()
        auc = roc_auc_score(y_true, y_pred)

        feat_entropies = np.concatenate([value for key, value in model.get_entropies().items() if 'data' in key])

        curr_agesex_ent = feat_entropies[1] + feat_entropies[2]
        if curr_agesex_ent > best_agesex_ent:
            best_agesex_ent = curr_agesex_ent
            best_sex_ent = feat_entropies[1]
            best_age_ent = feat_entropies[2]
            best_id = sample
        
        np.save(OUT_DIR + f'loss_history_id{sample}.npy', loss_history)
        model.to_npz(OUT_DIR + f'model_id{sample}.npz')

        df.loc[df.index.size] = [sample, final_epoch_loss, train_acc, 
                                    test_acc, auc, *feat_entropies]

        pbar.set_postfix_str(f'best entropy id {best_id}: age {best_age_ent:.2f}, sex {best_sex_ent:.2f}')
        pbar.update(1)

    pbar_train.close()
    pbar.close()
    df['sample'] = df['sample'].astype(int)
    df.to_csv(OUT_DIR + 'grid_search.csv', mode='a', header=not os.path.exists(OUT_DIR + 'grid_search.csv'))
    df_tot = pd.read_csv(OUT_DIR + 'grid_search.csv')
    df_tot.to_pickle(OUT_DIR + 'grid_search.pkl')

    with open(OUT_DIR + f'best_entropy{best_id}.txt', 'w') as f:
        f.write(f'best entropy id {best_id}: age {best_age_ent}, sex {best_sex_ent}')

if __name__ == '__main__':
    main()
