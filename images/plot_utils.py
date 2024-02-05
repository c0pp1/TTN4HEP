from itertools import combinations
import matplotlib.pyplot as plt

def plot_predictions(train_pred, test_pred, N_LABELS, FS=16, axs=None):
    combos = list(combinations(range(N_LABELS), 2))
    n_combos = len(combos)

    fig = None
    if axs is None:
        fig, axs = plt.subplots(2 if n_combos else 1, n_combos if n_combos else 1, figsize=(5*(n_combos if n_combos else 1), 8 if n_combos else 5))  
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
