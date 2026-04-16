# %%

import argparse
import os
import pickle as pkl
import torch as to
import numpy as np
import matplotlib.pyplot as plt
from qtorch import FixedPoint
from qtorch.quant import Quantizer
import tqdm

from ttnml.utils import dataloaders, accuracy, plot_roc_curves


class MPS(to.nn.Module):
    def __init__(
        self,
        phys_dim,
        n_sites,
        n_labels,
        max_bd,
        dtype=to.float32,
        quantizer=None,
        activation=None,
    ):

        super().__init__()
        self._num_sites = n_sites
        self._max_bd = max_bd
        self._n_labels = n_labels
        self._phys_dim = phys_dim

        self._label_idx = (self._num_sites - 1) // 2

        self._dtype = dtype
        self._quantizer = quantizer
        self.activation = activation

        self._tensors = (
            [to.rand(1, self._max_bd, self._phys_dim)]
            + [
                to.rand(self._max_bd, self._max_bd, self._phys_dim)
                for i in range(1, self._label_idx)
            ]
            + [to.rand(self._max_bd, self._max_bd, self._phys_dim, self._n_labels)]
            + [
                to.rand(self._max_bd, self._max_bd, self._phys_dim)
                for i in range(self._label_idx + 1, self._num_sites - 1)
            ]
            + [to.rand(self._max_bd, 1, self._phys_dim)]
        )

    def __getitem__(self, idx):
        return self._tensors[idx]

    def __len__(self):
        return self._num_sites

    @property
    def dtype(self):
        return self._dtype

    @property
    def n_labels(self):
        return self._n_labels

    @property
    def tensors(self):
        return self._tensors

    @classmethod
    def from_pickle(
        cls,
        filename: str,
        dtype=to.float32,
        quantizer=None,
        activation=None,
        c=1.0,
    ):
        with open(filename, "rb") as file:
            data = pkl.load(file)

        n_sites = len(data)
        max_bd = max(tens.shape[0] for tens in data)
        n_labels = data[n_sites // 2].shape[-1]
        phys_dim = data[0].shape[-1]

        obj = cls(
            phys_dim,
            n_sites,
            n_labels,
            max_bd,
            dtype=dtype,
            quantizer=quantizer,
            activation=activation,
        )
        obj._tensors = (
            [to.as_tensor(tens * c, dtype=dtype) for tens in data]
            if quantizer is None
            else [quantizer(to.as_tensor(tens * c, dtype=dtype)) for tens in data]
        )

        return obj

    def forward(self, x: to.Tensor, normalize=True, quantize=False, pbar=False):

        if pbar is None:
            pbar = tqdm.tqdm(total=self._num_sites, desc="Contracting...")
        # contract from left
        left = to.ones([x.shape[0], 1])
        for i in range(self._label_idx):
            site_tens = self._quantizer(x[:, i, ...]) if quantize else x[:, i, ...]
            left = to.einsum("ba,bd,acd->bc", left, site_tens, self[i])
            if quantize:
                left = self._quantizer(left)
            if pbar:
                pbar.update(1)

        # contract from right
        right = to.ones([x.shape[0], 1])
        for i in range(self._num_sites - 1, self._label_idx, -1):
            site_tens = self._quantizer(x[:, i, ...]) if quantize else x[:, i, ...]
            right = to.einsum("ba,bd,cad->bc", right, site_tens, self[i])
            if quantize:
                right = self._quantizer(right)
            if pbar:
                pbar.update(1)

        # contract the center
        site_tens = (
            self._quantizer(x[:, self._label_idx, ...])
            if quantize
            else x[:, self._label_idx, ...]
        )
        out = to.einsum(
            "ba,bd,bc,adce->be", left, right, site_tens, self[self._label_idx]
        )
        if quantize:
            out = self._quantizer(out)
        if pbar:
            pbar.update(1)
            pbar.close()
        if normalize:
            norm = to.norm(out, dim=-1, keepdim=True)
            out = out / (norm + 1e-15)
            out = self._quantizer(out) if quantize else out
        if self.activation is not None:
            out = self.activation(out)
        return self._quantizer(out) if quantize else out

    def predict(
        self,
        data: to.Tensor | to.utils.data.DataLoader,
        quantize=False,
        argmax=False,
    ):
        self.eval()
        predictions = []
        with to.no_grad():
            for x, _ in (
                data if isinstance(data, to.utils.data.DataLoader) else [(data, None)]
            ):
                x = x.to(dtype=self.dtype)
                out = self.forward(x, quantize=quantize)
                predictions.append(out.detach().cpu())
                if argmax:
                    predictions[-1] = to.argmax(predictions[-1], dim=-1)

        return to.cat(predictions, dim=0)


# %%

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Ema PTQ experiment")
    parser.add_argument(
        "-n",
        "--nconst",
        type=int,
        default=8,
    )
    args = parser.parse_args()

    NCONSTS = args.nconst
    # %%
    INT_BITS = 4
    FRAC_BITS = np.arange(2, 3)
    CS = [1.53]
    BATCH_SIZE = 1000
    DATASET = "hls150"
    MAPPING = "stacked_poly_mix"
    MAP_DIM = 2
    PERMUTATION = [4, 5, 13]
    SEEDS = [511, 123]
    to.manual_seed(SEEDS[0])

    results = np.empty((len(FRAC_BITS), len(CS), 3, 2))
    aucs = np.empty((len(FRAC_BITS), len(CS), 3, 5))
    fprs = np.empty((len(FRAC_BITS), len(CS), 3, 5))

    nthreads = os.environ.get("SLURM_CPUS_PER_TASK", 16)
    print(f"Using {nthreads} threads")
    to.set_num_threads(int(nthreads))

    # %%

    train_dl, test_dl, features = dataloaders.get_hls150_data_loaders(
        batch_size=BATCH_SIZE,
        mapping=MAPPING,
        dim=MAP_DIM,
        permutation=PERMUTATION,
        nconst=NCONSTS,
        norm="robust",
        transform=None,
        seed=SEEDS,
    )
    # %%
    pbar = tqdm.tqdm(total=len(FRAC_BITS) * len(CS) * 3, desc="Ema PTQ")
    for j, fl in enumerate(FRAC_BITS):
        wl = INT_BITS + fl
        for k, c in enumerate(CS):
            pbar.set_postfix({"FL": fl, "C": c})
            for fold in range(1, 4):

                model = MPS.from_pickle(
                    f"forAlberto/N={NCONSTS}/fold{fold}/model_weights.pkl",
                    activation=to.nn.Softmax(dim=-1),
                    c=c,
                    quantizer=Quantizer(FixedPoint(wl, fl), forward_rounding="nearest"),
                    dtype=to.float32,
                )
                model._tensors[model._label_idx] = (
                    model._tensors[model._label_idx] * 2.0
                )

                acc = accuracy(
                    model,
                    "cpu",
                    train_dl,
                    test_dl,
                    model.dtype,
                    quantize=True,
                    disable_pbar=True,
                )

                results[j, k, fold - 1, 0] = acc[0]
                results[j, k, fold - 1, 1] = acc[1]
                _, fprs_at_tpr, auc, roc_fig, roc_axs = plot_roc_curves(
                    model,
                    test_dl,
                    acc[1],
                    fold=fold,
                    quantize=True,
                )
                plt.close(roc_fig)
                aucs[j, k, fold - 1, :] = auc
                fprs[j, k, fold - 1, :] = fprs_at_tpr

                pbar.update(1)

            tqdm.tqdm.write(
                f"FL={fl}, C={c:.2f} => "
                f"\tTrain Acc: {results[j, k, :, 0].mean():.4f} ± {results[j, k, :, 0].std():.4f} | "
                f"Test Acc: {results[j, k, :, 1].mean():.4f} ± {results[j, k, :, 1].std():.4f}"
                "\n\t AUC: "
                + ", ".join(
                    [
                        f"{aucs[j, k, :, i].mean():.4f} ± {aucs[j, k, :, i].std():.4f}"
                        for i in range(3)
                    ]
                )
                + "\n\t FPR: "
                + ", ".join(
                    [
                        f"{fprs[j, k, :, i].mean():.4f} ± {fprs[j, k, :, i].std():.4f}"
                        for i in range(3)
                    ]
                )
            )
        # %%
        # np.save(
        #     f"forAlberto/N={NCONSTS}/results_fine2.part.npy",
        #     results,
        # )
    pbar.close()
    # np.save(
    #     f"forAlberto/N={NCONSTS}/results_fine2.npy",
    #     results,
    # )

    np.save(
        f"forAlberto/N={NCONSTS}/MPS{NCONSTS}_ptq_qops_aucs_best.npy",
        aucs.squeeze(),
    )
    np.save(
        f"forAlberto/N={NCONSTS}/MPS{NCONSTS}_ptq_qops_fprs_best.npy",
        fprs.squeeze(),
    )
