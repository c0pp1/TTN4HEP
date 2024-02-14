import ttn_torch as ttn
import torch
from qtorch.quant import fixed_point_quantize, Quantizer
from qtorch.optim import OptimLP
from qtorch import FixedPoint
import pandas as pd
from tqdm import tqdm
import numpy as np
from utils import accuracy, train_one_epoch, get_stripeimage_data_loaders, class_loss, get_titanic_data_loaders
import time
import traceback
from sklearn.metrics import roc_auc_score


DEVICE = 'cuda'
DEVICE = 'cuda' if torch.cuda.is_available() and DEVICE=='cuda' else 'cpu'
SCHEDULER_STEPS = 4
EPOCHS = 80
INIT_EPOCHS = 5
POPULATION = 10
LR = 0.02
LAMBDA = 0.1
DISABLE_PBAR = False
OUT_DIR = 'data/grid_search_fp2_titanic/'

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

def train_Q(model: ttn.TTNModel, weight_quant, train_dl, pbar = None, disable_pbar=False):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # turn your optimizer into a low precision optimizer
    optimizer = OptimLP(optimizer,
                        weight_quant=weight_quant)
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
    imsize = (4, 2)
    WLS = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    ILS = [1, 2]
    BATCH_SIZES = [128]
    QUANTIZE = [True, False]
    DTYPE = torch.float
    BOND_DIMS = [3, 8]

    df = pd.DataFrame(columns=['bond_dim', 'quantize', 'wl', 'il', 'sample', 'loss', 'train_acc', 'test_acc', 'train_acc0', 'test_acc0', 'auc', 'time'])

    pbar = tqdm(total=len(BOND_DIMS)*len(BATCH_SIZES)*(len(WLS)*len(ILS)+1)*POPULATION, position=0, desc='grid searching', disable=DISABLE_PBAR)
    pbar_train = tqdm(total=EPOCHS, position=1, desc='training', disable=DISABLE_PBAR)
    for bond_dim in BOND_DIMS:
        for batch_size in BATCH_SIZES:
            for quant in QUANTIZE:
                for wl in WLS if quant else [None]:
                    for il in ILS if quant else [None]:
                        for sample in range(POPULATION):

                            pbar.set_postfix_str(f'bd: {bond_dim}, bs: {batch_size}, quant: {quant}' + (f', wl: {wl}, il: {il}' if quant else '') + f', sample: {sample}')

                            train_dl, test_dl, feat = get_titanic_data_loaders(batch_size=batch_size, dtype=DTYPE, mapping='poly')
                            pbar_train.reset()
                            try:
                                if quant:
                                    
                                    # define the quantization parameters
                                    forward_num = FixedPoint(wl=wl, fl=wl-il)
                                    backward_num = FixedPoint(wl=wl, fl=wl-il)

                                    # Create a quantizer
                                    Q = Quantizer(forward_number=forward_num, backward_number=backward_num,
                                                    forward_rounding="nearest", backward_rounding="stochastic")
                                    
                                    weight_quant = lambda x : fixed_point_quantize(x, wl, wl-il, rounding="nearest")
                                    
                                    model = ttn.TTNModel(feat, bond_dim=bond_dim, n_labels=1, device=DEVICE, dtype=DTYPE, quantizer=Q)
                                else:
                                    model = ttn.TTNModel(feat, bond_dim=bond_dim, n_labels=1, device=DEVICE, dtype=DTYPE)

                                model.initialize(True, train_dl, class_loss, INIT_EPOCHS, disable_pbar=True)
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
                                with open(OUT_DIR + 'failed.txt', 'a') as f:
                                    f.write(f'feat: {feat}, bond_dim: {bond_dim}, quant: {quant}\n')
                                    f.write(str(e))
                                    f.write('\n')
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
                            
                            torch.save(model.state_dict(), OUT_DIR + f'model_bd{bond_dim}_q{quant}_wl{wl}_il{il}_id{sample}.pth')
                            np.save(OUT_DIR + f'loss_history_bd{bond_dim}_q{quant}_wl{wl}_il{il}_id{sample}.npy', loss_history)

                            df.loc[df.index.size] = [bond_dim, quant, wl, il, sample, final_epoch_loss, train_acc, 
                                                     test_acc, train_acc0, test_acc0, auc, end-start]

                            pbar.update(1)

    pbar_train.close()
    pbar.close()
    df.to_csv(OUT_DIR + 'grid_search.csv')
    df.to_pickle(OUT_DIR + 'grid_search.pkl')

if __name__ == '__main__':
    main()
