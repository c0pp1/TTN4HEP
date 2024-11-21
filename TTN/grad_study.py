from ttn_torch import *
from utils import *
import os

EPOCHS = 2
BS = 1000

def grad_search(dims, path = 'grad_study/'):
    for height, width in dims:
        print('Width:', width, 'Height:', height)

        train_dl, test_dl, features = get_stripeimage_data_loaders(height, width, batch_size=BS, mapping='spin', dim=2)

        model = TTNModel(features, 2, 1, dtype = torch.float64, device='cuda')
        model.initialize(True, train_dl, lambda *x: class_loss_fn(*x, l=0.1), epochs=5)

        optimizer = torch.optim.Adam(model.parameters(), 2**(model.n_layers-6))

        sweep_losses = []
        grads_magnitude = []
        for s in range(10):
            losses, grads = model.sweep(train_dl, dclass_loss_fn, optimizer, epochs=EPOCHS, path_type='layer', manual=True, loss=lambda*x: class_loss_fn(*x, l=0.))
            sweep_losses.extend(losses)
            grads_magnitude.extend(grads)

        train_acc, test_acc = accuracy(model, 'cuda', train_dl, test_dl, torch.float64)
        print('Train accuracy:', train_acc)
        print('Test accuracy:', test_acc)

        np.save(path + f'sweep_losses_{height}x{width}.npy', sweep_losses)
        np.save(path + f'grads_magnitude_{height}x{width}.npy', grads_magnitude)


if __name__ == '__main__':

    widths = [2, 2, 4, 8, 8, 16]
    heights = [2, 4, 4, 4, 8, 8]
    if not os.path.exists('grad_study/'):
        os.makedirs('grad_study/')
    grad_search(zip(heights, widths), path='grad_study/')
