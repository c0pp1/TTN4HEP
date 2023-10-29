import torch
import torchvision as tv
from quimb import tensor as qtn
import numpy as np
from tqdm.autonotebook import tqdm
from functools import partial

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


############# UTILS #############
#################################

def accuracy(model, device, train_dl, test_dl):
    correct = 0
    total = 0

    model.eval()
    model.to(device)

    with torch.no_grad():
        for data in tqdm(test_dl, total=len(test_dl), position=0, desc='test'):
            images, labels = data
            images, labels = images.to(device, dtype=torch.complex128).squeeze(), labels.to(device)
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
            images, labels = images.to(device, dtype=torch.complex128).squeeze(), labels.to(device)
            outputs = model(images)
            probs = torch.real(torch.pow(outputs, 2))
            probs = probs / torch.sum(probs)
            _, predicted = torch.max(probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_accuracy = correct / total

    return train_accuracy, test_accuracy