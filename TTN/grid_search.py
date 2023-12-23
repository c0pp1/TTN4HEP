import ttn_torch as ttn
import torch
import pandas as pd
from tqdm import tqdm
from utils import accuracy_binary, train_one_epoch_binary, get_stripeimage_data_loaders

DEVICE = 'cuda'
DEVICE = 'cuda' if torch.cuda.is_available() and DEVICE=='cuda' else 'cpu'
SCHEDULER_STEPS = 4
EPOCHS = 80
INIT_EPOCHS = 5
POPULATION = 10
LR = 0.05

def loss(labels, output):
    return torch.mean((output.squeeze() - labels)**2)/2

def train(model: ttn.TTNModel, train_dl, disable_pbar=False):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=-1, verbose=False), torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=1e-5)]
    scheduler = schedulers[0]
    tot_loss_history = []
    for epoch in range(EPOCHS):
        loss_history = train_one_epoch_binary(model, DEVICE, train_dl, loss, optimizer, pbar=pbar, disable_pbar=disable_pbar)
        tot_loss_history += loss_history
        if epoch % SCHEDULER_STEPS == SCHEDULER_STEPS-1:
            scheduler.step()

    loss_history = np.array(tot_loss_history)
    print('Accuracy on train and test set:', accuracy_binary(model, DEVICE, train_dl, test_dl, DTYPE, disable_pbar=True))

    weights = [tensor.detach().cpu().flatten() for tensor in model.tensors]
    weights_ls.append(torch.concat(weights, dim=0))
    train_dl, test_dl, h = get_iris_data_loaders(batch_size=BATCH_SIZE, labels=['Iris-virginica', 'Iris-versicolor'])
    return torch.stack(weights_ls)


def main():

    FEATURES = [4, 8, 16, 32, 64]
    BATCH_SIZES = [8, 32, 128, 512]
    INITIALIZE = [True, False]
    DTYPES = [torch.double, torch.cdouble]

    df = pd.DataFrame(columns=['features', 'bond_dim', 'batch_size', 'initialize', 'dtype'])

    train_dl, test_dl, h = get_iris_data_loaders(batch_size=BATCH_SIZE, labels=['Iris-virginica', 'Iris-versicolor'])

    pbar = tqdm(total=EPOCHS*len(train_dl)*len(FEATURES)*len(BATCH_SIZES)*len(INITIALIZE)*len(DTYPES)*POPULATION, position=0, desc='grid searching', leave=True)
    for feat in FEATURES:        
        model = TTNModel(features, bond_dim=BOND_DIM, n_labels=1, device=DEVICE, dtype=DTYPE)
        model.initialize(True, train_dl, loss, INIT_EPOCHS, disable_pbar=True)
        model.train()
        model.to(DEVICE)