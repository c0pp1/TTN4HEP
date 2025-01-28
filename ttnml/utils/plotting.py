from itertools import combinations
import colorsys
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk

__all__ = [
    "plot_predictions",
    "plot_confusion_matrix",
    "plot_loss",
    "plot_feat_en",
    "adjust_brightness",
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
    y_pred, y_true, classes, FS=14, cmap=plt.cm.Blues, thresh=0.5, ax=None
):

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

    cm = sk.metrics.confusion_matrix(y_true, y_pred)

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_xticks(np.arange(len(classes)), classes, fontsize=FS - 2)
    ax.set_yticks(np.arange(len(classes)), classes, fontsize=FS - 2)
    ax.set_xlabel("Predicted label", fontsize=FS)
    ax.set_ylabel("True label", fontsize=FS)
    ax.set_title("Confusion matrix", fontsize=FS + 2)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = "d"
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
    imp, names, dataset, map_dim=2, labels=[""], FS=14, axs=None, color="tab:blue"
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
        ax.barh(names, imp[:, i], color=color)
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
