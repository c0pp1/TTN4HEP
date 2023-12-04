import torch
import torchvision as tv
from quimb import tensor as qtn
import numpy as np
from tqdm.autonotebook import tqdm
from functools import partial

from algebra import sep_contract_torch

############# DATASET HANDLING #############
############################################

# this function takes an image as a square matrix and flattens it
# by flattening 2x2 blocks of pixels into a 4 pixels vector
def linearize(tensor: torch.Tensor):
    result = torch.clone(tensor).reshape((-1, np.prod(tensor.shape[-2:])))
    index = torch.as_tensor(range(result.shape[-1]))
    mask = index // 2 % 2 == 0

    for i in range(tensor.shape[-1]):
        result[
            :,
            (mask != i % 2)
            & (index < (i // 2 + 1) * 2 * tensor.shape[-1])
            & (index >= (i // 2) * 2 * tensor.shape[-1]),
        ] = tensor[:, i, :]

    return result

# quantum embedding
def quantize(tensor):
    cos = torch.cos(torch.pi * tensor / 2)
    sin = torch.sin(torch.pi * tensor / 2)

    return torch.stack([cos, sin], dim=-1)


def load_to_device(tensor: torch.Tensor, device):
    return tensor.to(device)


# this function balances the specified labels in the train and test sets
# it assumes that the labels are integers
# labels: list of labels to balance
# train: training set
# test: test set
# returns: balanced train and test sets
def balance(labels, train, test):

    # get the number of samples in each class
    train_class_counts = np.bincount(train.targets)
    test_class_counts = np.bincount(test.targets)

    # get the maximum number of samples in a class
    train_max_class_count = min(train_class_counts[labels])
    test_max_class_count = min(test_class_counts[labels])

    # get the indices of the samples in each class
    train_indices = [np.where(np.array(train.targets) == label)[0] for label in labels]
    test_indices = [np.where(np.array(test.targets) == label)[0] for label in labels]

    # get the indices of the samples in each class
    # that will be used for training
    train_indices_balanced = [
        np.random.choice(train_index, train_max_class_count)
        for train_index in train_indices
    ]
    test_indices_balanced = [
        np.random.choice(test_index, test_max_class_count)
        for test_index in test_indices
    ]

    # get the balanced training and test sets
    train_balanced = torch.utils.data.Subset(
        train, np.concatenate(train_indices_balanced)
    )
    test_balanced = torch.utils.data.Subset(test, np.concatenate(test_indices_balanced))

    return train_balanced, test_balanced


def get_ttn_transform(h, device="cpu"):
    return tv.transforms.Compose(
        [
            tv.transforms.Resize((h, h)),
            tv.transforms.ToTensor(),
            tv.transforms.Lambda(partial(load_to_device, device=device)),
            tv.transforms.Lambda(linearize),
            tv.transforms.Lambda(quantize),
        ]
    )


def get_ttn_transform_visual(h):
    return tv.transforms.Compose(
        [tv.transforms.Resize((h, h)), tv.transforms.ToTensor()]
    )

def get_mnist_data_loaders(h, batch_size, labels=[0, 1], device="cpu", path='../data'):
    # get the training and test sets
    train = tv.datasets.MNIST(
        root=path, train=True, download=True, transform=get_ttn_transform(h, device)
    )
    test = tv.datasets.MNIST(
        root=path, train=False, download=True, transform=get_ttn_transform(h, device)
    )
    train_visual = tv.datasets.MNIST(
        root=path, download=True, train=True, transform=get_ttn_transform_visual(h)
    )

    # balance the training and test sets
    train_balanced, test_balanced = balance(labels, train, test)

    NUM_WORKERS = torch.get_num_threads()

    test_dl = torch.utils.data.DataLoader(test_balanced, batch_size=batch_size)
    train_dl = torch.utils.data.DataLoader(train_balanced, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    train_visual = torch.utils.data.DataLoader(train_visual, batch_size=batch_size)

    return train_dl, test_dl, train_visual


############# UTILS #############
#################################

def accuracy(model, device, train_dl, test_dl, dtype=torch.complex128):
    correct = 0
    total = 0

    model.eval()
    model.to(device)

    with torch.no_grad():
        for data in tqdm(test_dl, total=len(test_dl), position=0, desc='test'):
            images, labels = data
            images, labels = images.to(device, dtype=dtype).squeeze(), labels.to(device)
            outputs = model(images)
            probs = torch.real(torch.pow(outputs, 2))
            probs = probs / torch.sum(probs)
            _, predicted = torch.max(probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_accuracy = correct / total

        correct = 0
        total = 0

        for data in tqdm(train_dl, total=len(train_dl), position=0, desc='train'):
            images, labels = data
            images, labels = images.to(device, dtype=dtype).squeeze(), labels.to(device)
            outputs = model(images)
            probs = torch.real(torch.pow(outputs, 2))
            probs = probs / torch.sum(probs)
            _, predicted = torch.max(probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_accuracy = correct / total

    return train_accuracy, test_accuracy

def one_epoch_one_tensor(tensor, data_tn_batched, train_dl, optimizer, loss_fn, device='cuda', pbar=None):
    # perform one epoch of optimization of a single tensor
    # given the data_tn and the optimizer
    tot_data = 0
    lossess = []
    if pbar is None:
        pbar = tqdm(data_tn_batched, total=len(data_tn_batched),position=0)
    with torch.autograd.set_detect_anomaly(True):
        for data_tn, batch in zip(data_tn_batched, train_dl):
            optimizer.zero_grad()
            labels = batch[1].to(device=device)
            data = torch.stack([tensor.data for tensor in data_tn['l1']])
            outputs = sep_contract_torch([tensor], data)
            
            probs = torch.real(torch.pow(outputs, 2))
            probs = probs / torch.sum(probs)
            loss = loss_fn(labels, probs)

            loss.backward()
            optimizer.step()
            lossess.append(loss.cpu())
            tot_data += labels.shape[0]
            pbar.update()
            pbar.set_postfix_str(f'loss: {loss.item():.3f}')
    
    pbar.set_postfix({'loss': loss.item(), 'epoch mean loss': np.array([loss.item() for loss in lossess]).sum() / tot_data})
    pbar.close()
    return lossess



############# GRAPHICS #############
####################################
import colorsys
from matplotlib import colors
def adjust_lightness(color, amount=0.5):
    try:
        c = colors.cnames[color]
    except:
        c = color
    rgb = len(c) == 3
    c_hls = colorsys.rgb_to_hls(*colors.to_rgb(c))
    return colorsys.hls_to_rgb(c_hls[0], max(0, min(1, amount * c_hls[1])), c_hls[2]) + ((c[3],) if not rgb else ())