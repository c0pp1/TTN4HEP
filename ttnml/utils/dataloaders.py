import os
from functools import partial
from math import floor
import numpy as np
import pandas as pd
import torch
import torchvision as tv
from scipy.io.arff import loadarff
from sklearn.preprocessing import MinMaxScaler

from .embeddings import embeddings_dict

__all__ = [
    "get_mnist_data_loaders",
    "get_higgs_data_loaders",
    "get_iris_data_loaders",
    "get_titanic_data_loaders",
    "get_bb_data_loaders",
    "get_fsoco_data_loaders",
    "get_hls_data_loaders",
    "get_stripeimage_data_loaders",
]

module_dir = os.path.dirname(os.path.abspath(__file__))


# this function takes an image as a square matrix and flattens it
# by flattening 2x2 blocks of pixels into a 4 pixels vector
def linearize(tensor: torch.Tensor):
    height, width = tensor.shape[-2:]
    result = torch.zeros_like(tensor).view(-1, height * width)

    for i in range(height):
        for j in range(width):
            new_index = ((i // 2) * (width // 2) + (j // 2)) * 4
            if i % 2 != 0:
                new_index += 2
            if j % 2 != 0:
                new_index += 1
            result[:, new_index % (height * width)] = tensor[:, i, j]

    return result


def load_to_device(tensor: torch.Tensor, device):
    return tensor.to(device)


# this function balances the specified labels in the train and test sets
# it assumes that the labels are integers
# labels: list of labels to balance
# train: training set
# test: test set
# returns: balanced train and test sets
def balance(labels, samples, sel_labels=None):

    labels_unique = np.unique(labels)
    if sel_labels is not None:
        labels_unique = sel_labels

    # get the number of samples in each class
    class_counts = np.bincount(labels)

    # get the maximum number of samples in a class
    max_class_count = min(class_counts[labels_unique])

    # get the indices of the samples in each class
    indices = [np.where(np.array(labels) == label)[0] for label in labels_unique]

    # get the indices of the samples in each class
    # that will be used for training
    indices_balanced = np.stack(
        [
            np.random.choice(train_index, max_class_count, replace=False)
            for train_index in indices
        ],
        axis=1,
    )

    """
    # get the balanced training and test sets
    balanced = torch.utils.data.Subset(
        samples, np.concatenate(indices_balanced)
    )
    """
    return samples[indices_balanced.flatten()], labels[indices_balanced.flatten()]


def get_ttn_transform(h, mapping: str, dim=2, device="cpu"):
    return tv.transforms.Compose(
        [
            tv.transforms.Resize((h, h), antialias=True),
            tv.transforms.Lambda(partial(load_to_device, device=device)),
            tv.transforms.Lambda(linearize),
            tv.transforms.Lambda(partial(embeddings_dict[mapping], dim=dim)),
        ]
    )


def get_ttn_transform_visual(h):
    return tv.transforms.Resize((h, h))


def get_mnist_data_loaders(
    h,
    batch_size,
    labels=[0, 1],
    dtype=torch.double,
    mapping: str = "spin",
    dim=2,
    device="cpu",
    path=os.path.join(module_dir, "../../data"),
):
    # get the training and test sets
    train = tv.datasets.MNIST(
        root=path,
        train=True,
        download=True,
        transform=get_ttn_transform(h, mapping, dim, device),
    )
    test = tv.datasets.MNIST(
        root=path,
        train=False,
        download=True,
        transform=get_ttn_transform(h, mapping, dim, device),
    )
    train_visual = tv.datasets.MNIST(
        root=path, download=True, train=True, transform=get_ttn_transform_visual(h)
    )
    # TODO: avoid using dataset.data as you lose transformations
    # balance the training and test sets
    train_balanced, train_labels = balance(train.targets, train.data, labels)
    test_balanced, test_labels = balance(test.targets, test.data, labels)

    train_balanced = torch.utils.data.TensorDataset(
        get_ttn_transform(h, mapping, dim, device)(train_balanced).to(dtype=dtype),
        train_labels.to(dtype=dtype),
    )
    test_balanced = torch.utils.data.TensorDataset(
        get_ttn_transform(h, mapping, dim, device)(test_balanced).to(dtype=dtype),
        test_labels.to(dtype=dtype),
    )

    NUM_WORKERS = torch.get_num_threads()

    test_dl = torch.utils.data.DataLoader(test_balanced, batch_size=batch_size)
    train_dl = torch.utils.data.DataLoader(
        train_balanced,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )
    train_visual = torch.utils.data.DataLoader(train_visual, batch_size=batch_size)

    return train_dl, test_dl, train_visual, h * h


def get_higgs_data_loaders(
    batch_size,
    dtype=torch.double,
    mapping: str = "spin",
    dim=2,
    device="cpu",
    path=os.path.join(module_dir, "../../data"),
    permutation=None,
):
    # get the training and test sets
    train = torch.tensor(np.load(path + "/Higgs/higgs_train.npy"))
    test = torch.tensor(np.load(path + "/Higgs/higgs_test.npy"))
    train_labels = torch.tensor(
        np.load(path + "/Higgs/higgs_train_labels.npy"), dtype=torch.float64
    )
    test_labels = torch.tensor(
        np.load(path + "/Higgs/higgs_test_labels.npy"), dtype=torch.float64
    )

    if permutation is not None:
        train = train[:, permutation]
        test = test[:, permutation]

    train = embeddings_dict[mapping](load_to_device(train, device), dim=dim).to(
        dtype=dtype
    )
    test = embeddings_dict[mapping](load_to_device(test, device), dim=dim).to(
        dtype=dtype
    )

    train = torch.utils.data.TensorDataset(train, train_labels)
    test = torch.utils.data.TensorDataset(test, test_labels)

    NUM_WORKERS = torch.get_num_threads()
    return (
        torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
        ),
        torch.utils.data.DataLoader(test, batch_size=batch_size),
        8,
    )


def get_stripeimage_data_loaders(
    h,
    w,
    batch_size,
    dtype=torch.double,
    mapping: str = "spin",
    dim=2,
    device="cpu",
    path=os.path.join(module_dir, "../../data"),
):
    # get the training and test sets
    train = torch.tensor(np.load(path + f"/stripeimages/{h}x{w}train.npy"))
    test = torch.tensor(np.load(path + f"/stripeimages/{h}x{w}test.npy"))
    train_labels = torch.tensor(
        np.load(path + f"/stripeimages/{h}x{w}train_labels.npy"), dtype=torch.float64
    )
    test_labels = torch.tensor(
        np.load(path + f"/stripeimages/{h}x{w}test_labels.npy"), dtype=torch.float64
    )

    train = embeddings_dict[mapping](
        linearize(load_to_device(train, device)), dim=dim
    ).to(dtype=dtype)
    test = embeddings_dict[mapping](
        linearize(load_to_device(test, device)), dim=dim
    ).to(dtype=dtype)

    train = torch.utils.data.TensorDataset(train, train_labels)
    test = torch.utils.data.TensorDataset(test, test_labels)
    NUM_WORKERS = torch.get_num_threads()
    return (
        torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
        ),
        torch.utils.data.DataLoader(test, batch_size=batch_size),
        h * w,
    )


def get_iris_data_loaders(
    batch_size,
    sel_labels=["Iris-setosa", "Iris-versicolor"],
    dtype=torch.double,
    mapping: str = "spin",
    dim=2,
    device="cpu",
    path=os.path.join(module_dir, "../../data"),
    permutation=None,
):
    dataframe = pd.read_csv(path + "/iris/Iris.csv")
    dataframe = dataframe.sample(frac=1)

    dataframe["SepalLengthCm"] = (
        dataframe["SepalLengthCm"] / dataframe["SepalLengthCm"].max()
    )
    dataframe["SepalWidthCm"] = (
        dataframe["SepalWidthCm"] / dataframe["SepalWidthCm"].max()
    )
    dataframe["PetalLengthCm"] = (
        dataframe["PetalLengthCm"] / dataframe["PetalLengthCm"].max()
    )
    dataframe["PetalWidthCm"] = (
        dataframe["PetalWidthCm"] / dataframe["PetalWidthCm"].max()
    )

    dataframe = dataframe[dataframe["Species"].isin(sel_labels)]
    np.save(path + f"/iris/iris_index.npy", dataframe.index.to_numpy())
    dataframe = dataframe.reset_index(drop=True)
    labels = dataframe["Species"].to_numpy()
    labels = np.unique(labels, return_inverse=True)[1]
    np.save(path + f"/iris/iris_labels.npy", labels)
    labels = torch.nn.functional.one_hot(torch.tensor(labels), len(sel_labels))

    data = dataframe.drop(columns=["Id", "Species"]).to_numpy()
    np.save(path + f"/iris/iris_normalized.npy", data)
    if permutation is not None:
        data = data[:, permutation]
    data = torch.tensor(data)

    data = embeddings_dict[mapping](load_to_device(data, device), dim=dim).to(
        dtype=dtype
    )
    train_size = int(0.8 * len(data))
    np.save(path + f"/iris/iris_embedded.npy", data)
    train = torch.utils.data.TensorDataset(
        data[:train_size], labels[:train_size].to(dtype=torch.float64)
    )
    test = torch.utils.data.TensorDataset(
        data[train_size:], labels[train_size:].to(dtype=torch.float64)
    )
    NUM_WORKERS = torch.get_num_threads()
    return (
        torch.utils.data.DataLoader(
            train, batch_size=batch_size, num_workers=NUM_WORKERS
        ),
        torch.utils.data.DataLoader(test, batch_size=batch_size),
        4 if permutation is None else len(permutation),
    )


def get_titanic_data_loaders(
    batch_size,
    dtype=torch.double,
    mapping: str = "spin",
    dim=2,
    device="cpu",
    path=os.path.join(module_dir, "../../data"),
    scale=(0, 1),
    permutation=None,
):
    dataframe = pd.read_csv(path + "/titanic/titanic.csv")
    # dataframe = dataframe.sample(frac=1)
    dataframe["sex"] = pd.Categorical(dataframe["sex"]).codes
    dataframe["embarked"] = pd.Categorical(dataframe["embarked"]).codes

    # transform not numerical ticket codes to numerical
    mask = np.array([not x.isdigit() for x in dataframe["ticket"]])
    dataframe.loc[mask, "ticket"] = pd.Categorical(dataframe["ticket"][mask]).codes
    dataframe["ticket"] = dataframe["ticket"].astype(int)

    labels = torch.tensor(dataframe["survived"].to_numpy())

    # scale the data
    scaler = MinMaxScaler(scale)
    df_scaled = pd.DataFrame(scaler.fit_transform(dataframe.drop(columns=["survived"])))

    # embed
    data = torch.tensor(df_scaled.to_numpy())
    if permutation is not None:
        data = data[:, permutation]
    data = embeddings_dict[mapping](load_to_device(data, device), dim=dim).to(
        dtype=dtype
    )

    # balance the training and test sets
    data_balanced, labels_balanced = balance(labels, data)
    train_size = int(0.8 * len(data_balanced))

    train = torch.utils.data.TensorDataset(
        data_balanced[:train_size], labels_balanced[:train_size].to(dtype=torch.float64)
    )
    test = torch.utils.data.TensorDataset(
        data_balanced[train_size:], labels_balanced[train_size:].to(dtype=torch.float64)
    )

    NUM_WORKERS = torch.get_num_threads()
    return (
        torch.utils.data.DataLoader(
            train, batch_size=batch_size, num_workers=NUM_WORKERS
        ),
        torch.utils.data.DataLoader(test, batch_size=batch_size),
        8 if permutation is None else len(permutation),
    )


titanic_features = np.array(
    ["pclass", "sex", "age", "sibsp", "parch", "ticket", "fare", "embarked"]
)


def get_bb_data_loaders(
    batch_size,
    dtype=torch.double,
    mapping: str = "spin",
    dim=2,
    device="cpu",
    path=os.path.join(module_dir, "../../data"),
    scale=(0, 1),
    permutation=None,
):
    train_set = np.loadtxt(path + "/bbdata/asym_train.csv", delimiter=",")
    test_set = np.loadtxt(path + "/bbdata/asym_test.csv", delimiter=",")
    # dataframe = dataframe.sample(frac=1)
    train_labels = (torch.tensor(train_set[:, 0], dtype=torch.float64) + 1) / 2
    test_labels = (torch.tensor(test_set[:, 0], dtype=torch.float64) + 1) / 2
    # scale the data
    scaler = MinMaxScaler(scale)
    train_scaled = scaler.fit_transform(train_set[:, 1:])
    test_scaled = scaler.transform(test_set[:, 1:])

    train_data = torch.tensor(train_scaled)
    test_data = torch.tensor(test_scaled)
    if permutation is not None:
        train_data = train_data[:, permutation]
        test_data = test_data[:, permutation]
    train_data = embeddings_dict[mapping](
        load_to_device(train_data, device), dim=dim
    ).to(dtype=dtype)
    test_data = embeddings_dict[mapping](load_to_device(test_data, device), dim=dim).to(
        dtype=dtype
    )

    train_data = torch.utils.data.TensorDataset(train_data, train_labels)
    test_data = torch.utils.data.TensorDataset(test_data, test_labels)

    NUM_WORKERS = torch.get_num_threads()
    return (
        torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, num_workers=NUM_WORKERS
        ),
        torch.utils.data.DataLoader(test_data, batch_size=batch_size),
        16 if permutation is None else len(permutation),
    )


def get_fsoco_data_loaders(
    batch_size,
    dtype=torch.double,
    device="cpu",
    path=os.path.join(module_dir, "../../data"),
    permutation=None,
):
    data = np.load(path + "/fsoco/patches_32x32.npy").reshape(-1, 32 * 32, 4)
    labels = np.load(path + "/fsoco/labels_32x32.npy")

    if permutation is not None:
        data = data[:, permutation]

    train_size = floor(0.8 * len(data))
    train_data = torch.tensor(data[:train_size])
    test_data = torch.tensor(data[train_size:])
    train_labels = torch.tensor(labels[:train_size], dtype=torch.float64)
    test_labels = torch.tensor(labels[train_size:], dtype=torch.float64)

    train_data = load_to_device(train_data, device).to(dtype=dtype)
    test_data = load_to_device(test_data, device).to(dtype=dtype)

    train_data = torch.utils.data.TensorDataset(train_data, train_labels)
    test_data = torch.utils.data.TensorDataset(test_data, test_labels)

    NUM_WORKERS = torch.get_num_threads()
    return (
        torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, num_workers=NUM_WORKERS
        ),
        torch.utils.data.DataLoader(test_data, batch_size=batch_size),
        32 * 32,
    )


def get_hls_data_loaders(
    batch_size,
    dtype=torch.double,
    mapping: str = "spin",
    dim=2,
    device="cpu",
    path=os.path.join(module_dir, "../../data"),
    scale=(0, 1),
    permutation=None,
    sel_labels=None,
):

    raw_data = loadarff(path + "/hls4ml_jets/hls4ml_HLF.arff")
    dataframe = pd.DataFrame(raw_data[0])
    dataframe = dataframe.sample(frac=1)
    dataframe["class"] = dataframe["class"].str.decode("utf-8")
    if sel_labels is not None:
        dataframe = dataframe[dataframe["class"].isin(sel_labels)]
    dataframe["class"] = pd.Categorical(dataframe["class"]).codes

    labels = torch.tensor(dataframe["class"].to_numpy(), dtype=torch.int64)
    # scale the data
    scaler = MinMaxScaler(scale)
    df_scaled = pd.DataFrame(scaler.fit_transform(dataframe.drop(columns=["class"])))

    # embed
    data = torch.tensor(df_scaled.to_numpy())
    if permutation is not None:
        data = data[:, permutation]
    data = embeddings_dict[mapping](load_to_device(data, device), dim=dim).to(
        dtype=dtype
    )

    # balance the training and test sets
    data_balanced, labels_balanced = balance(labels, data)

    labels_balanced = torch.nn.functional.one_hot(
        labels_balanced, len(np.unique(labels_balanced))
    )
    train_size = int(0.8 * len(data_balanced))

    train = torch.utils.data.TensorDataset(
        data_balanced[:train_size], labels_balanced[:train_size].to(dtype=dtype)
    )
    test = torch.utils.data.TensorDataset(
        data_balanced[train_size:], labels_balanced[train_size:].to(dtype=dtype)
    )

    NUM_WORKERS = torch.get_num_threads()
    return (
        torch.utils.data.DataLoader(
            train, batch_size=batch_size, num_workers=NUM_WORKERS
        ),
        torch.utils.data.DataLoader(test, batch_size=batch_size),
        16 if permutation is None else len(permutation),
    )
