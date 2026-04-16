from itertools import combinations
import colorsys
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import os

__all__ = [
    "plot_predictions",
    "plot_confusion_matrix",
    "plot_loss",
    "plot_feat_en",
    "adjust_brightness",
    "plot_roc_curves",
    "plot_correlations",
]


def plot_predictions(
    train_pred,
    test_pred,
    N_LABELS,
    FS=14,
    nbins=50,
    train_true=None,
    test_true=None,
    axs=None,
):
    combos = list(combinations(range(N_LABELS), 2))
    n_combos = len(combos)

    fig = None
    if axs is None:
        fig, axs = plt.subplots(
            2 if n_combos else 1,
            n_combos if n_combos else 1,
            figsize=(6 * (n_combos if n_combos else 1), 10 if n_combos else 5),
        )
        axs = axs if n_combos == 0 else axs.flatten()

    if n_combos:
        for i, (c1, c2) in enumerate(combos):
            axs[i].hist2d(
                train_pred[:, c1].numpy(), train_pred[:, c2].numpy(), bins=nbins
            )
            axs[i].set_xlabel(f"{c1} prediction", fontsize=FS)
            axs[i].set_ylabel(f"{c2} prediction", fontsize=FS)
            axs[i].set_title(
                f"Predictions distribution train {c1} vs {c2}", fontsize=FS + 2
            )

            axs[i + n_combos].hist2d(
                test_pred[:, c1].numpy(), test_pred[:, c2].numpy(), bins=nbins
            )
            axs[i + n_combos].set_xlabel(f"{c1} prediction", fontsize=FS)
            axs[i + n_combos].set_ylabel(f"{c2} prediction", fontsize=FS)
            axs[i + n_combos].set_title(
                f"Predictions distribution test {c1} vs {c2}", fontsize=FS + 2
            )

    elif train_true is None and test_true is None:
        axs.hist(
            [train_pred, test_pred],
            bins=nbins,
            label=["train", "test"],
            stacked=True,
            edgecolor="white",
        )
        axs.legend()
        axs.set_xlabel("Prediction", fontsize=FS)
        axs.set_ylabel("Counts", fontsize=FS)
        axs.set_title("Predictions distribution", fontsize=FS + 2)
    else:
        n1, bins1, patches1 = axs.hist(
            [train_pred[train_true == 0], test_pred[test_true == 0]],
            bins=nbins,
            label=["0", "0"],
            color=["tab:blue", "tab:blue"],
            stacked=True,
            edgecolor="white",
            alpha=0.6,
        )
        n2, bins2, patches2 = axs.hist(
            [train_pred[train_true == 1], test_pred[test_true == 1]],
            bins=nbins,
            label=["1", "1"],
            color=["tab:orange", "tab:orange"],
            stacked=True,
            edgecolor="white",
            alpha=0.6,
        )

        for patch in patches1[1]:
            patch.set_hatch("//")
        for patch in patches2[1]:
            patch.set_hatch("\\\\")
        axs.legend()
        axs.set_xlabel("Prediction", fontsize=FS)
        axs.set_ylabel("Counts", fontsize=FS)
        axs.set_title("Predictions distribution", fontsize=FS + 2)

    return fig, axs


def plot_confusion_matrix(
    y_pred,
    y_true,
    classes,
    FS=14,
    cmap=plt.cm.Blues,
    thresh=0.5,
    ax=None,
    normalize="pred",
    fmt="d",
):
    from sklearn.metrics import confusion_matrix

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        fig = None

    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    else:
        y_pred = y_pred > thresh
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=-1)

    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_xticks(np.arange(len(classes)), classes, fontsize=FS - 2)
    ax.set_yticks(np.arange(len(classes)), classes, fontsize=FS - 2)
    ax.set_xlabel("Predicted label", fontsize=FS)
    ax.set_ylabel("True label", fontsize=FS)
    ax.set_title("Confusion matrix", fontsize=FS + 2)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    cthresh = (cm.max() - cm.min()) / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > (cm.min() + cthresh) else "black",
                fontsize=FS - 2,
            )
    return fig, ax


def plot_loss(losses, ax, epochs, FS=14, sweep=None):
    steps_per_epoch = len(losses) // epochs
    if sweep is not None:

        if sweep:
            ax.plot(
                [
                    np.array(losses[i : i + steps_per_epoch]).mean()
                    for i in np.arange(0, len(losses), steps_per_epoch)
                ]
            )
            ax.set_xticks(np.linspace(0, epochs, 13, dtype=int))
            ax.set_xlabel("Sweep", fontsize=FS)
        else:
            ax.plot(losses)
            ax.set_xlabel("Step", fontsize=FS)
    else:

        ax.plot(
            [
                losses[
                    i
                    * steps_per_epoch : (
                        (i + 1) * steps_per_epoch if i < epochs - 1 else None
                    )
                ].mean()
                for i in range(epochs)
            ]
        )
        ax.set_xticks(np.linspace(0, epochs, 11, dtype=int))
        ax.set_xlabel("Epoch", fontsize=FS)

    ax.set_ylabel("Loss", fontsize=FS)
    ax.tick_params(axis="both", which="major", labelsize=FS - 2)
    ax.grid(axis="y")

    ax.hlines(
        losses[(epochs - 1) * steps_per_epoch :].mean(),
        0,
        epochs,
        colors="r",
        linestyles="dashed",
    )
    ax.text(
        epochs,
        losses[(epochs - 1) * steps_per_epoch :].mean() + 0.001,
        f"{losses[(epochs-1)*steps_per_epoch:].mean():.3f}",
        fontsize=FS - 2,
        color="r",
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    return ax


def plot_feat_en(
    imp,
    names,
    dataset,
    map_dim=2,
    labels=[""],
    yerr=None,
    FS=14,
    axs=None,
    color="tab:blue",
):
    unit = int(len(names) ** 0.5) + 2
    max_entropy = np.log(map_dim)
    if axs is None:
        fig, axs = plt.subplots(
            1,
            len(labels),
            figsize=(unit * len(labels) + (len(labels) == 1), unit),
            sharey=True,
            sharex=True,
        )
    else:
        fig = None

    imp = np.stack(imp)
    axs = np.array(axs).flatten()
    for i, ax in enumerate(axs):
        ax.barh(names, imp[:, i], color=color, xerr=yerr, capsize=3)
        ax.set_xlabel("Entropy", fontsize=FS)
        ax.set_title(labels[i], fontsize=FS + 2)

        ax.tick_params(axis="both", which="major", labelsize=FS - 2)

    axs[0].set_xlim(0, max_entropy + 0.05)
    axs[0].set_ylabel("Feature", fontsize=FS)

    for ax in axs:
        ax.axvline(max_entropy, color="tab:red", linestyle="--")
    if fig is not None:
        fig.suptitle(f"{dataset} dataset features entropy", fontsize=FS + 4)
    return fig, axs


def find_nearest(array: np.ndarray, value: float):
    """Finds the index of the nearest value in an array to a given value."""
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def plot_roc_curves(
    model,
    test_dl,
    total_acc: float,
    labels: list[str] = ["Gluon", "Quark", "W", "Z", "Top"],
    colors: list[str] = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"],
    fold=None,
    quantize=False,
):
    """Plot two types of ROC curves: standard and flipped."""

    y_test = np.concatenate([y for _, y in test_dl], axis=0)
    y_pred = model.predict(test_dl, quantize=quantize).numpy()
    if y_test.ndim == 1:
        y_test = np.expand_dims(y_test, axis=-1)
        y_pred = np.expand_dims(y_pred, axis=-1)

    from sklearn.metrics import roc_curve, auc

    tpr_baseline = np.linspace(0.025, 0.99, 100)

    fprs, aucs, fprs_at_tpr = [], [], []

    # Create figure with 2 subplots in one row
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Loop over each class
    for idx, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_test[:, idx], y_pred[:, idx])
        auc_value = auc(fpr, tpr)
        aucs.append(auc_value)

        # Interpolate FPR for a fixed set of TPR points
        fpr_baseline = np.interp(tpr_baseline, tpr, fpr)
        fprs.append(fpr_baseline)

        # Find the closest TPR to 60% and get corresponding FPR
        tpr_idx = find_nearest(tpr, 0.8)
        fprs_at_tpr.append(fpr[tpr_idx])

        # Left subplot (Standard ROC Curve: TPR vs FPR)
        axs[0].plot(
            fpr,
            tpr,
            color=colors[idx],
            label=f"{label}: AUC={auc_value*100:.2f}%; FPR @ 80% TPR={fprs_at_tpr[-1]:.3f}",
        )

        # Right subplot (Flipped ROC Curve: FPR vs TPR)
        axs[1].plot(fpr, tpr, color=colors[idx], label=f"{label}")

    # Formatting Left Plot (Standard ROC)
    axs[0].set_xlabel("False Positive Rate (FPR)")
    axs[0].set_ylabel("True Positive Rate (TPR)")
    axs[0].set_ylim(0.001, 1)
    axs[0].semilogy()  # Log scale for better visualization
    fold_suffix = f" - Fold {fold}" if fold is not None else ""
    axs[0].set_title(f"ROC Curves (Total Acc: {total_acc:.4f}%){fold_suffix}")
    axs[0].legend()

    # Formatting Right Plot (Flipped ROC)
    axs[1].set_ylabel("True Positive Rate (TPR)")
    axs[1].set_xlabel("False Positive Rate (FPR)")
    axs[1].set_ylim(0.001, 1)
    axs[1].set_title("ROC Curve (nolog)")
    axs[1].legend()

    fig.tight_layout()

    return fprs, fprs_at_tpr, aucs, fig, axs


def plot_correlations(corr, features, labels, fig=None, axs=None, FS=12, annot=True):
    n_feat = len(features)
    if axs is None:
        fig, axs = plt.subplots(1, len(labels), figsize=(n_feat * len(labels), n_feat))
    axs = np.atleast_1d(axs)
    for i, ax in enumerate(axs):
        im = ax.imshow(
            corr[:, :, i].T, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal"
        )
        ax.set_xticks(np.arange(n_feat), features, fontsize=FS, rotation=45)
        ax.set_yticks(np.arange(n_feat), features, fontsize=FS)
        # Minor ticks
        ax.set_xticks(np.arange(-0.5, n_feat, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_feat, 1), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
        # ax.set_xticklabels(features)
        # ax.set_yticklabels(features)
        ax.set_title(labels[i], fontsize=FS + 2)
        if annot:
            for n in range(n_feat):
                for m in range(n_feat):
                    if not np.isnan(corr[m, n, i]):
                        text = ax.text(
                            m,
                            n,
                            round(corr[m, n, i], 2),
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=FS - 2,
                        )
    return fig, axs, im


############# GRAPHICS #############
####################################


def adjust_brightness(color, amount=0.5):

    try:
        c = colors.cnames[color]
    except:
        c = color

    rgb = len(c) == 3
    c_hls = colorsys.rgb_to_hls(*colors.to_rgb(c))
    return colorsys.hls_to_rgb(
        c_hls[0], max(0, min(1, amount * c_hls[1])), c_hls[2]
    ) + ((c[3],) if not rgb else ())
