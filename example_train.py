import argparse
from functools import partial
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
import numpy as np
from datetime import datetime
from time import perf_counter
import os
import json
import sys
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from ttnml.ml import TTNModel
from ttnml.tn import check_correct_init
from ttnml.utils import *
from torchinfo import summary

from tqdm import tqdm

FONTSIZE = 14
slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
slurm_cpus = int(slurm_cpus) if slurm_cpus else (os.cpu_count() - 1)
torch.set_num_threads(slurm_cpus)
print(f"Using {torch.get_num_threads()} threads.")
SLURM_ID = os.getenv("SLURM_JOB_ID")

# define json structure
results = {"hyperparams": {"dataset": {}, "model": {}, "training": {}}, "results": {}}

############################
###### SELECT DATASET ######
############################

parser = argparse.ArgumentParser()
parser.add_argument("--bd", type=int, default=10, help="Bond dimension")
parser.add_argument(
    "--map",
    type=str,
    default="spin",
    help="Mapping to use: 'spin', 'poly', 'stacked_poly'",
)
parser.add_argument("--map-dim", type=int, default=2, help="Dimension of the input map")
parser.add_argument(
    "--kfolds", type=int, default=3, help="Number of k-folds for cross-validation"
)
parser.add_argument("--nconst", type=int, default=None, help="Number of constituents")
args = parser.parse_args()

h = 8
BATCH_SIZE = 1000
DATASET = "hls150"
MAPPING = args.map
MAP_DIM = args.map_dim
FEATURES = [4, 5, 13]
NCONST = args.nconst
NORM = "robust"
TRANSFORM = "log10->5"
ONE_HOT = False
map_kwargs = (
    {}
)  # {'n_part_per_site': 3, 'part_per_feat': [np.arange(12), np.arange(36)]}
dataset_params = results["hyperparams"]["dataset"]
dataset_params["name"] = DATASET
dataset_params["batch_size"] = BATCH_SIZE
dataset_params["mapping"] = MAPPING
dataset_params["map_dim"] = MAP_DIM
dataset_params["map_kwargs"] = map_kwargs
dataset_params["nconst"] = NCONST
dataset_params["features"] = FEATURES
dataset_params["norm"] = NORM
dataset_params["transform"] = TRANSFORM
dataset_params["one_hot"] = ONE_HOT

iris_features = ["SL", "SW", "PL", "PW"]

if DATASET == "mnist":
    train_dl, test_dl, train_visual, n_features = get_mnist_data_loaders(
        h, batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM, labels=[0, 1]
    )
elif DATASET == "stripe":
    train_dl, test_dl, n_features = get_stripeimage_data_loaders(
        4, h, batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM
    )
elif DATASET == "iris":
    # worst performance with iris-versicolor and iris-virginica
    train_dl, test_dl, features = get_iris_data_loaders(
        batch_size=BATCH_SIZE,
        sel_labels=["Iris-setosa", "Iris-versicolor"],
        mapping=MAPPING,
        dim=MAP_DIM,
    )
elif DATASET == "higgs":
    train_dl, test_dl, n_features = get_higgs_data_loaders(
        batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM
    )
elif DATASET == "titanic":
    train_dl, test_dl, n_features = get_titanic_data_loaders(
        batch_size=BATCH_SIZE,
        scale=(0, 1),
        mapping=MAPPING,
        dim=MAP_DIM,
        permutation=FEATURES,
    )  # scales different from (0, 1) are reasonable only in the poly mapping
elif DATASET == "bbdata":
    train_dl, test_dl, n_features = get_bb_data_loaders(
        batch_size=BATCH_SIZE,
        mapping=MAPPING,
        dim=MAP_DIM,
        permutation=FEATURES,
        one_hot=ONE_HOT,
    )  # permutation=[0,1,5,7,10,12,14,15]
elif DATASET == "hls":
    train_dl, test_dl, n_features = get_hls_data_loaders(
        batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM
    )
elif DATASET == "hls150":
    train_dl, test_dl, n_features = get_hls150_data_loaders(
        batch_size=BATCH_SIZE,
        mapping=MAPPING,
        dim=MAP_DIM,
        permutation=FEATURES,
        transform=TRANSFORM,
        nconst=NCONST,
        norm=NORM,
    )  # , map_kwargs={'n_part_per_site': 3, 'part_per_feat': [np.arange(12), np.arange(36)]}
else:
    raise ValueError(f"Unknown dataset: {DATASET}")

dataset_params["training_size"] = len(train_dl.dataset)
dataset_params["test_size"] = len(test_dl.dataset)

############################
### SELECT MODEL PARAMS ####
############################

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BOND_DIM = args.bd
DTYPE = torch.double
dtype_eps = torch.finfo(DTYPE).eps
MODEL_DIR = f"trained_models/{DATASET}_models/data"
(features, n_phys), label_shape = (x.shape[-2:] for x in next(iter(test_dl)))
N_LABELS = 1 if len(label_shape) == 1 else label_shape[-1]

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

model_params = results["hyperparams"]["model"]
model_params["bond_dim"] = BOND_DIM
model_params["phys_dim"] = n_phys
model_params["dtype"] = str(DTYPE)
model_params["device"] = DEVICE
model_params["label_dim"] = N_LABELS


##########################
######## TRAINING ########
##########################

INITIALIZE = False
LR = 0.001
GAMMA = 0.9
EPOCHS = 100
gauging = False
LOSS_PARAMS = {"l": 1e-4}
SCHEDULER_STEPS = 5
STOP_AFTER = 100
LOSS_FN = class_loss_fn
LOSS = partial(LOSS_FN, **LOSS_PARAMS)
KFOLDS = args.kfolds
OPTIMIZER = torch.optim.Adam
SCHEDULER = torch.optim.lr_scheduler.ExponentialLR
training_params = results["hyperparams"]["training"]
training_params["initialize"] = INITIALIZE
training_params["optimizer"] = {OPTIMIZER.__name__: {"lr": LR}}
training_params["scheduler"] = {
    SCHEDULER.__name__: {"gamma": GAMMA, "step_size": SCHEDULER_STEPS}
}
training_params["loss"] = {str(LOSS_FN): LOSS_PARAMS}
training_params["epochs"] = EPOCHS
training_params["gauging"] = gauging
training_params["kfolds"] = KFOLDS

if KFOLDS > 1:
    kfold = KFold(n_splits=KFOLDS, shuffle=True)
    dataset = ConcatDataset([train_dl.dataset, test_dl.dataset])
    NUM_WORKERS = int(np.ceil(torch.get_num_threads() / 5))
    fold_iterator = [
        (
            fold,
            (
                DataLoader(
                    Subset(dataset, train_ids),
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                ),
                DataLoader(
                    Subset(dataset, test_ids),
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                ),
            ),
        )
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset))
    ]
else:
    fold_iterator = [(0, (train_dl, test_dl))]


now = datetime.now()
results["hyperparams"]["date"] = now.strftime("%Y%m%d-%H%M%S")
folds_train_accs = []
folds_test_accs = []
folds_aucs = []
folds_fprs_at_tpr = []
for fold, (train_dl, test_dl) in fold_iterator:
    print(f"\n############# Fold {fold+1} #############\n")
    start = perf_counter()
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

    INIT_EPOCHS = 20
    loss = lambda *x: class_loss_fn(*x, l=0.01)
    # loss = ClassLoss(0.1, transform=torch.tanh)

    print("Initializing the model...", end=" ")
    model.initialize(INITIALIZE, train_dl, loss, INIT_EPOCHS, disable_pbar=True)
    print("done \U00002714")
    # if not INITIALIZE:
    #     # gauge the network
    #     model.canonicalize("0.0")
    print(check_correct_init(model, atol=1e-6))

    summary(
        model, input_size=(BATCH_SIZE, features, n_phys), dtypes=[DTYPE], device=DEVICE
    )

    model.to(DEVICE)
    optimizer = OPTIMIZER(model.parameters(), lr=LR)
    scheduler = SCHEDULER(optimizer, GAMMA, last_epoch=-1)

    tot_loss_history = []
    train_accs = []
    test_accs = []
    epoch_pbar = tqdm(range(EPOCHS), desc="Training...", total=EPOCHS, file=sys.stdout)
    for epoch in epoch_pbar:
        model.train()
        loss_history = train_one_epoch(
            model, DEVICE, train_dl, LOSS, optimizer, gauging=gauging, disable_pbar=True
        )
        tot_loss_history += loss_history
        epoch_pbar.set_postfix(loss=loss_history[-1])

        if (epoch % SCHEDULER_STEPS == SCHEDULER_STEPS - 1) and (epoch < STOP_AFTER):
            scheduler.step()
            # pass

        model.eval()
        acc = accuracy(model, DEVICE, train_dl, test_dl, model.dtype, disable_pbar=True)
        train_accs.append(acc[0])
        test_accs.append(acc[1])

    end = perf_counter()
    print(f"Training time: {end - start:.2f} seconds")

    loss_history = np.array(tot_loss_history)
    train_accs = np.array(train_accs)
    test_accs = np.array(test_accs)

    ##########################
    ## EVALUATION AND PLOTS ##
    ##########################

    print(f"Train accuracy: {train_accs[-1]}")
    print(f"Test accuracy: {test_accs[-1]}")
    print(f"Train loss: {loss_history[-1]}")
    folds_train_accs.append(train_accs[-1])
    folds_test_accs.append(test_accs[-1])

    _, fprs_at_tpr, auc, roc_fig, roc_axs = plot_roc_curves(
        model,
        test_dl,
        test_accs[-1],
    )
    folds_aucs.append(auc)
    folds_fprs_at_tpr.append(fprs_at_tpr)

    results["results"][f"fold_{fold}"] = {
        "train_accuracy": train_accs[-1],
        "test_accuracy": test_accs[-1],
        "train_loss": loss_history[-1],
        "training_time": end - start,
        "aucs": auc,
        "fprs_at_tpr": fprs_at_tpr,
    }

    accs_fig, accs_ax = plt.subplots(figsize=(6, 4))
    accs_ax.plot(train_accs, label="Train")
    accs_ax.plot(test_accs, label="Test")
    accs_ax.set_xlabel("Epoch", fontsize=FONTSIZE)
    accs_ax.set_ylabel("Accuracy", fontsize=FONTSIZE)
    accs_ax.tick_params(axis="both", which="major", labelsize=FONTSIZE - 2)
    accs_ax.legend(fontsize=FONTSIZE)
    accs_fig.tight_layout()

    loss_fig, loss_ax = plt.subplots(1, 1, figsize=(6, 4))
    loss_ax = plot_loss(loss_history, loss_ax, EPOCHS, FS=FONTSIZE)
    loss_ax.set_title(f"Training Loss\n{DATASET}, BD={BOND_DIM}", fontsize=FONTSIZE + 2)
    # ax.set_ylim(1150, 1240)
    loss_fig.tight_layout()

    ######################
    ####### SAVING #######
    ######################

    print(f"Saving to {MODEL_DIR}...")
    print(
        f"\tLoss history: loss_history{'_fold' + str(fold) if KFOLDS > 1 else ''}_{SLURM_ID}.npy",
        end=" ",
    )
    np.save(
        MODEL_DIR
        + f"/loss_history{'_fold' + str(fold) if KFOLDS > 1 else ''}_{SLURM_ID}.npy",
        loss_history,
    )
    print("✔")
    print(
        f"\tAccuracies: accuracies{'_fold' + str(fold) if KFOLDS > 1 else ''}_{SLURM_ID}.npy",
        end=" ",
    )
    np.save(
        MODEL_DIR
        + f"/accuracies{'_fold' + str(fold) if KFOLDS > 1 else ''}_{SLURM_ID}.npy",
        np.stack([train_accs, test_accs], axis=-1),
    )
    print("✔")
    print(
        f"\tModel: model_{DATASET}_bd{BOND_DIM}_{MAPPING}{'_fold' + str(fold) if KFOLDS > 1 else ''}_{SLURM_ID}.npz",
        end=" ",
    )
    model.to_npz(
        MODEL_DIR
        + f"/model_{DATASET}_bd{BOND_DIM}_{MAPPING}{'_fold' + str(fold) if KFOLDS > 1 else ''}_{SLURM_ID}.npz"
    )
    print("✔")
    print("\tPlots:")
    print(
        f"\t\tAccuracies: accuracies{'_fold' + str(fold) if KFOLDS > 1 else ''}_{SLURM_ID}.pdf",
        end=" ",
    )
    accs_fig.savefig(
        MODEL_DIR
        + f"/accuracies{'_fold' + str(fold) if KFOLDS > 1 else ''}_{SLURM_ID}.pdf"
    )
    print("✔")
    print(
        f"\t\tLoss: loss_history{'_fold' + str(fold) if KFOLDS > 1 else ''}_{SLURM_ID}.pdf",
        end=" ",
    )
    loss_fig.savefig(
        MODEL_DIR
        + f"/loss_history{'_fold' + str(fold) if KFOLDS > 1 else ''}_{SLURM_ID}.pdf"
    )
    print("✔")
    print(
        f"\t\tROCs: roc_curve{'_fold' + str(fold) if KFOLDS > 1 else ''}_{SLURM_ID}.pdf",
        end=" ",
    )
    roc_fig.savefig(
        os.path.join(
            MODEL_DIR,
            f"roc_curve{'_fold' + str(fold) if KFOLDS > 1 else ''}_{SLURM_ID}.pdf",
        )
    )
    print("✔")

model_params["trainable_params"] = sum(p.numel() for p in model.tensors)

folds_train_accs = np.array(folds_train_accs)
folds_test_accs = np.array(folds_test_accs)
folds_aucs = np.array(folds_aucs)
folds_fprs_at_tpr = np.array(folds_fprs_at_tpr)

results["results"]["avg_train_acc"] = folds_train_accs.mean()
results["results"]["avg_test_acc"] = folds_test_accs.mean()
results["results"]["std_train_acc"] = folds_train_accs.std()
results["results"]["std_test_acc"] = folds_test_accs.std()
results["results"]["avg_aucs"] = folds_aucs.mean(0).tolist()
results["results"]["std_aucs"] = folds_aucs.std(0).tolist()
results["results"]["avg_fprs_at_tpr"] = folds_fprs_at_tpr.mean(0).tolist()
results["results"]["std_fprs_at_tpr"] = folds_fprs_at_tpr.std(0).tolist()

print("Saving results json...", end=" ")
with open(MODEL_DIR + f"/../results_{SLURM_ID}.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)
print("All done ✔")
