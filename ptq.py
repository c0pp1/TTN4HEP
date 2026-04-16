# %%
import argparse
from glob import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from qtorch import FixedPoint
from qtorch.quant import Quantizer, fixed_point_quantize

from ttnml import TTNModel
from ttnml.utils import *

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) - 1)
print(f"Using {torch.get_num_threads()} threads for PTQ.")

# %%
if __name__ == "__main_":
    parser = argparse.ArgumentParser(description="Post-training quantization")
    parser.add_argument(
        "-n", "--nconst", type=int, default=16, help="Number of constituents"
    )
    args = parser.parse_args()
else:

    class Args:
        nconst = 8

    args = Args()


# %%
def load_model(fname: str, **kwargs) -> list[tuple[TTNModel, int] | tuple[TTNModel]]:
    """Load a TTNModel from file.

    Args:
        fname (str): Path to the results json file.
        kwargs: Additional keyword arguments to pass to TTNModel

    Returns:
        TTNModel: The loaded TTNModel instance. If kfold is greater than 1, returns a list of TTNModel instances.
    """

    slurm_id = fname.split("_")[-1].split(".")[0]
    with open(fname, "r") as f:
        config = json.load(f)
    kfold = config.get("hyperparams", {}).get("training", {}).get("kfold", 1)
    wls = config.get("hyperparams", {}).get("training", {}).get("WL", False)

    if kfold > 1:
        print("Loading k-fold models...")
    if wls:
        print("QAT detected, returning list of models...")

    models = []
    for model_fname in glob(os.path.dirname(fname) + f"/data/model_*_{slurm_id}*.npz"):
        if "iso" in model_fname:
            continue
        quantizer = kwargs.pop("quantizer", None)
        if quantizer is None and wls:
            wl = int(model_fname.split("_")[-2].split("wl")[-1])
            fl = int(model_fname.split("_")[-1].split("fl")[-1][:-4])
            forward_num = FixedPoint(wl=wl, fl=fl)
            backward_num = FixedPoint(wl=wl, fl=fl)
            quantizer = Quantizer(
                forward_number=forward_num,
                backward_number=backward_num,
                forward_rounding="nearest",
                backward_rounding="nearest",
            )

        model = TTNModel.from_npz(model_fname, quantizer=quantizer, **kwargs)
        models.append((model, fl) if wls else (model,))
    return models


def get_dataset(results: dict, **kwargs):
    dataset_params: dict = results.get("hyperparams", {}).get("dataset", {})
    name = dataset_params.get("name", "mnist").lower()
    MAPPING = kwargs.get("mapping", dataset_params.get("mapping", "stacked_poly"))
    MAP_DIM = kwargs.get("map_dim", dataset_params.get("map_dim", 2))
    BATCH_SIZE = kwargs.get("batch_size", dataset_params.get("batch_size", 500))
    FEATURES = kwargs.get("features", dataset_params.get("features", None))
    NCONST = kwargs.get("nconst", dataset_params.get("nconst", 16))
    NORM = kwargs.get("norm", dataset_params.get("norm", "robust"))
    h = kwargs.get("h", 4)
    if name == "mnist":
        train_dl, test_dl, train_visual, n_features = get_mnist_data_loaders(
            h, batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM, labels=[0, 1]
        )
        return train_dl, test_dl, n_features
    if name == "stripe":
        train_dl, test_dl, n_features = get_stripeimage_data_loaders(
            4, h, batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM
        )
        return train_dl, test_dl, n_features
    if name == "iris":
        # worst performance with iris-versicolor and iris-virginica
        train_dl, test_dl, features = get_iris_data_loaders(
            batch_size=BATCH_SIZE,
            sel_labels=["Iris-setosa", "Iris-virginica", "Iris-versicolor"],
            mapping=MAPPING,
            dim=MAP_DIM,
        )
        return train_dl, test_dl, features
    if name == "higgs":
        train_dl, test_dl, n_features = get_higgs_data_loaders(
            batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM
        )
        return train_dl, test_dl, n_features
    if name == "titanic":
        train_dl, test_dl, n_features = get_titanic_data_loaders(
            batch_size=BATCH_SIZE, scale=(0, 1), mapping=MAPPING, dim=MAP_DIM
        )  # scales different from (0, 1) are reasonable only in the poly mapping
        return train_dl, test_dl, n_features
    if name == "bbdata":
        train_dl, test_dl, n_features = get_bb_data_loaders(
            batch_size=BATCH_SIZE,
            mapping=MAPPING,
            dim=MAP_DIM,
            permutation=FEATURES,
        )  # permutation=[0,1,5,7,10,12,14,15]
        return train_dl, test_dl, n_features
    if name == "hls":
        train_dl, test_dl, n_features = get_hls_data_loaders(
            batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM
        )
        return train_dl, test_dl, n_features
    if name == "hls150":
        train_dl, test_dl, n_features = get_hls150_data_loaders(
            batch_size=BATCH_SIZE,
            mapping=MAPPING,
            dim=MAP_DIM,
            permutation=FEATURES,
            nconst=NCONST,
            norm=NORM,
            transform="log10->4",
        )  # , map_kwargs={'n_part_per_site': 3, 'part_per_feat': [np.arange(12), np.arange(36)]}

        return train_dl, test_dl, n_features

    raise ValueError(f"Unknown dataset: {name}")


# %%
FLS = list(range(2, 17))
WLS = list(range(4, 19))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
slurm_ids = {32: 179826, 16: 179819, 8: 194424}
# 32 179826
# 16 179819
# 8 194424
SLURM_ID = slurm_ids[args.nconst]
results_fname = f"/shared/home/coppi/repositories/tn4hep/TTN4HEP/trained_models/hls150_models_robust/results_{SLURM_ID}.json"

results = {}
with open(results_fname, "r") as f:
    results = json.load(f)

train_dl, test_dl, _ = get_dataset(
    results,
    dtype=torch.float32,
)

models = load_model(results_fname, dtype=torch.float32)

test_mean, test_std, auc_mean, auc_std, fpr_mean, fpr_std = [], [], [], [], [], []
for fl, wl in zip(FLS, WLS):
    print(f"Quantization with WL={wl}, FL={fl}")
    quantizer = Quantizer(
        forward_number=FixedPoint(wl=wl, fl=fl),
        backward_number=FixedPoint(wl=wl, fl=fl),
        forward_rounding="nearest",
        backward_rounding="nearest",
    )

    accs = []
    folds_aucs = []
    folds_fprs_at_tpr = []
    for fold, (model,) in enumerate(models):
        q_model = TTNModel.from_ttn(model, quantizer=quantizer)
        q_model.tensors = [
            fixed_point_quantize(t, wl=wl, fl=fl, rounding="nearest")
            for t in q_model.tensors
        ]

        q_model.initialize()
        q_model.eval()
        train_acc, test_acc = accuracy(
            q_model,
            DEVICE,
            train_dl,
            test_dl,
            q_model.dtype,
            disable_pbar=True,
            quantize=True,
        )
        _, fprs_at_tpr, auc, roc_fig, roc_axs = plot_roc_curves(
            q_model,
            test_dl,
            test_acc,
            fold=fold,
            quantize=True,
        )
        plt.close(roc_fig)
        folds_aucs.append(auc)
        folds_fprs_at_tpr.append(fprs_at_tpr)
        accs.append(test_acc)
    test_mean.append(np.mean(accs))
    test_std.append(np.std(accs))
    auc_mean.append(np.mean(folds_aucs, axis=0))
    auc_std.append(np.std(folds_aucs, axis=0))
    fpr_mean.append(np.mean(folds_fprs_at_tpr, axis=0))
    fpr_std.append(np.std(folds_fprs_at_tpr, axis=0))
    print(f"Average test accuracy: {test_mean[-1]:.4f} ± {test_std[-1]:.4f}")
    print(
        "Average AUC: "
        + ", ".join(
            f"{a_m:.4f} ± {a_s:.4f}" for a_m, a_s in zip(auc_mean[-1], auc_std[-1])
        )
    )
    print(
        f"Average FPR at TPR=0.8: "
        + ", ".join(
            f"{fpr_m:.4f} ± {fpr_s:.4f}"
            for fpr_m, fpr_s in zip(fpr_mean[-1], fpr_std[-1])
        )
    )
    print("-" * 50)

print("Saving results...")
# if os.path.exists(
#     os.path.dirname(results_fname) + f"/ptq_qop_test_mean_{SLURM_ID}.npy"
# ):
#     previous_means = np.load(
#         os.path.dirname(results_fname) + f"/ptq_qop_test_mean_{SLURM_ID}.npy"
#     ).tolist()
#     previous_stds = np.load(
#         os.path.dirname(results_fname) + f"/ptq_qop_test_std_{SLURM_ID}.npy"
#     ).tolist()
#     test_mean = previous_means + test_mean
#     test_std = previous_stds + test_std
# np.save(
#     os.path.dirname(results_fname) + f"/ptq_qop_test_mean_{SLURM_ID}.npy",
#     np.array(test_mean),
# )
# np.save(
#     os.path.dirname(results_fname) + f"/ptq_qop_test_std_{SLURM_ID}.npy",
#     np.array(test_std),
# )

np.save(
    os.path.dirname(results_fname) + f"/ptq_qop_auc_mean_{SLURM_ID}.npy",
    np.array(auc_mean),
)
np.save(
    os.path.dirname(results_fname) + f"/ptq_qop_auc_std_{SLURM_ID}.npy",
    np.array(auc_std),
)
np.save(
    os.path.dirname(results_fname) + f"/ptq_qop_fpr_mean_{SLURM_ID}.npy",
    np.array(fpr_mean),
)
np.save(
    os.path.dirname(results_fname) + f"/ptq_qop_fpr_std_{SLURM_ID}.npy",
    np.array(fpr_std),
)
print("Done.")

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

slurm_ids = {32: 179606, 16: 179601, 8: 179595}
nconst = 32
qat_dirname = "/shared/home/coppi/repositories/tn4hep/TTN4HEP/trained_models/hls150_robust_quantmodels"
results_fname = os.path.join(qat_dirname, f"results_{slurm_ids[nconst]}.json")
with open(results_fname, "r") as f:
    results = json.load(f)

models = load_model(results_fname, dtype=torch.float32)
train_dl, test_dl, _ = get_dataset(results, dtype=torch.float32)
test_accs = []
fls = []
for model, fl in models:
    model.initialize()
    model.eval()
    train_acc, test_acc = accuracy(
        model,
        DEVICE,
        train_dl,
        test_dl,
        model.dtype,
        disable_pbar=True,
        quantize=False,
    )
    test_accs.append(test_acc)
    fls.append(fl)
    print(f"FL={fl}: Test accuracy: {test_acc:.4f}")
test_accs = sorted(test_accs, key=lambda x: fls[test_accs.index(x)])
np.save(
    os.path.join(qat_dirname, f"qat_fpop_test_accs_{slurm_ids[nconst]}.npy"),
    np.array(test_accs),
)
print("Done.")
# %%
