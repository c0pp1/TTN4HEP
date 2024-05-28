from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(train_pred, test_pred, N_LABELS, FS=16, axs=None):
    combos = list(combinations(range(N_LABELS), 2))
    n_combos = len(combos)

    fig = None
    if axs is None:
        fig, axs = plt.subplots(2 if n_combos else 1, n_combos if n_combos else 1, figsize=(6*(n_combos if n_combos else 1), 10 if n_combos else 5))  
        axs = axs if n_combos == 0 else axs.flatten()

    if n_combos:
        for i, (c1, c2) in enumerate(combos):
            axs[i].hist2d(train_pred[:, c1].numpy(), train_pred[:, c2].numpy(), bins=50)
            axs[i].set_xlabel(f'{c1} prediction')
            axs[i].set_ylabel(f'{c2} prediction')
            axs[i].set_title(f'Predictions distribution train {c1} vs {c2}')

            axs[i+n_combos].hist2d(test_pred[:, c1].numpy(), test_pred[:, c2].numpy(), bins=50)
            axs[i+n_combos].set_xlabel(f'{c1} prediction')
            axs[i+n_combos].set_ylabel(f'{c2} prediction')
            axs[i+n_combos].set_title(f'Predictions distribution test {c1} vs {c2}')

    else:
        axs.hist([train_pred, test_pred], bins=50, label=['train', 'test'], stacked=True, edgecolor='white')
        axs.legend()
        axs.set_xlabel('Prediction', fontsize=FS)
        axs.set_ylabel('Counts', fontsize=FS)
        axs.set_title('Predictions distribution', fontsize=FS+2)

    if axs is None:
        fig.tight_layout()

    return fig, axs


def plot_loss(losses, ax, epochs, FS=14, sweep=None):

    if sweep is not None:
        if sweep:
            ax.plot([np.array(losses[i:i+len(losses)//epochs]).mean() for i in np.arange(0, len(losses), len(losses)//epochs)])
            ax.set_xticks(np.arange(0, epochs))
            ax.set_xlabel('Sweep', fontsize=FS)
        else:
            ax.plot(losses)
            ax.set_xlabel('Step', fontsize=FS)
        ax.set_ylabel('Loss', fontsize=FS)
        ax.tick_params(axis='both', which='major', labelsize=FS-2)
        ax.grid(axis='y')
        return ax

    steps_per_epoch = len(losses) // epochs

    ax.plot([losses[i*steps_per_epoch:((i+1)*steps_per_epoch if i<epochs-1 else None)].mean() for i in range(epochs)])
    ax.set_xlabel('Epoch', fontsize=FS)
    ax.set_ylabel('Loss', fontsize=FS)
    ax.tick_params(axis='both', which='major', labelsize=FS-2)
    ax.grid(axis='y')

    ax.hlines(losses[(epochs-1)*steps_per_epoch:].mean(), 0, epochs, colors='r', linestyles='dashed')
    ax.text(epochs, losses[(epochs-1)*steps_per_epoch:].mean()+0.001, f'{losses[(epochs-1)*steps_per_epoch:].mean():.3f}', fontsize=FS-2, color='r', verticalalignment='bottom', horizontalalignment='right')

    return ax