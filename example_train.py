import torch
import numpy as np
from datetime import datetime
import os

from ttnml.ml import TTNModel
from ttnml.tn import check_correct_init, TIndex
from ttnml.tn.algebra import contract_up
from ttnml.utils import *
from torchinfo import summary

from tqdm.notebook import tqdm, trange

FONTSIZE = 14
torch.set_num_threads(30)

############################
###### SELECT DATASET ######
############################

h = 8
features = h**2
BATCH_SIZE = 1000
DATASET = "hls150"
MAPPING = "stacked_poly"
MAP_DIM = 2

iris_features = ["SL", "SW", "PL", "PW"]

if DATASET == "mnist":
    train_dl, test_dl, train_visual, features = get_mnist_data_loaders(
        h, batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM, labels=[0, 1]
    )
elif DATASET == "stripe":
    train_dl, test_dl, features = get_stripeimage_data_loaders(
        4, h, batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM
    )
elif DATASET == "iris":
    # worst performance with iris-versicolor and iris-virginica
    train_dl, test_dl, features = get_iris_data_loaders(
        batch_size=BATCH_SIZE,
        sel_labels=["Iris-setosa", "Iris-virginica", "Iris-versicolor"],
        mapping=MAPPING,
        dim=MAP_DIM,
    )
elif DATASET == "higgs":
    train_dl, test_dl, features = get_higgs_data_loaders(
        batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM
    )
elif DATASET == "titanic":
    train_dl, test_dl, features = get_titanic_data_loaders(
        batch_size=BATCH_SIZE, scale=(0, 1), mapping=MAPPING, dim=MAP_DIM
    )  # scales different from (0, 1) are reasonable only in the poly mapping
elif DATASET == "bbdata":
    train_dl, test_dl, features = get_bb_data_loaders(
        batch_size=BATCH_SIZE,
        mapping=MAPPING,
        dim=MAP_DIM,
        permutation=[0, 1, 5, 7, 10, 12, 13, 15],
    )  # permutation=[0,1,5,7,10,12,14,15]
elif DATASET == "hls":
    train_dl, test_dl, features = get_hls_data_loaders(
        batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM
    )
elif DATASET == "hls150":
    train_dl, test_dl, features = get_hls150_data_loaders(
        batch_size=BATCH_SIZE,
        mapping=MAPPING,
        dim=MAP_DIM,
        permutation=[4, 5, 13],
        nconst=16,
        norm="minmax",
    )  # , map_kwargs={'n_part_per_site': 3, 'part_per_feat': [np.arange(12), np.arange(36)]}
else:
    raise ValueError(f"Unknown dataset: {DATASET}")


############################
### SELECT MODEL PARAMS ####
############################

DEVICE = "cpu" if torch.cuda.is_available() else "cpu"
BOND_DIM = 10
N_LABELS = 5
DTYPE = torch.double
dtype_eps = torch.finfo(DTYPE).eps
MODEL_DIR = f"trained_models/{DATASET}_models_minmax"
features, n_phys = next(iter(train_dl))[0].shape[-2:]
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

model = TTNModel(
    features,
    n_phys=n_phys,
    bond_dim=BOND_DIM,
    n_labels=N_LABELS,
    device=DEVICE,
    dtype=DTYPE,
)

##########################
#### INITIALIZE MODEL ####
##########################

INIT_EPOCHS = 5
loss = lambda *x: class_loss_fn(*x, l=0.01)
# loss = ClassLoss(0.1, transform=torch.tanh)

model.initialize(True, train_dl, loss, INIT_EPOCHS)
print(check_correct_init(model, atol=1e-6))
summary(model, input_size=(BATCH_SIZE, features, n_phys), dtypes=[DTYPE], device=DEVICE)


##########################
######## TRAINING ########
##########################

LR = 0.001
EPOCHS = 50
gauging = False
LAMBDA = 1e-4
SCHEDULER_STEPS = 5
LOSS = lambda *x: class_loss_fn(*x, l=LAMBDA)

model.train()
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, 0.9, last_epoch=-1, verbose=False
)

tot_loss_history = []
now = datetime.now()
for epoch in trange(EPOCHS, desc="epochs", position=0):
    loss_history = train_one_epoch(
        model, DEVICE, train_dl, LOSS, optimizer, gauging=gauging
    )
    tot_loss_history += loss_history

    if epoch % SCHEDULER_STEPS == SCHEDULER_STEPS - 1:
        scheduler.step()
        # pass

loss_history = np.array(tot_loss_history)

###########################
## EVALUATION AND SAVING ##
###########################
model.eval()

acc = accuracy(model, DEVICE, train_dl, test_dl, model.dtype)
print(f"Train accuracy: {acc[0]}")
print(f"Test accuracy: {acc[1]}")
print(f"Train loss: {loss_history[-1]}")

print(
    f"Saving to {MODEL_DIR}/model_{DATASET}_bd{BOND_DIM}_{MAPPING}_{now.strftime('%Y%m%d-%H%M%S')}.npz"
)

np.save(MODEL_DIR + f'/loss_history_{now.strftime("%Y%m%d-%H%M%S")}.npy', loss_history)
model.to_npz(
    MODEL_DIR
    + f'/model_{DATASET}_bd{BOND_DIM}_{MAPPING}_{now.strftime("%Y%m%d-%H%M%S")}.npz'
)
