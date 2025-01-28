import torch
import torchvision as tv
from quimb import tensor as qtn
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from tqdm import tqdm
from functools import partial
from sklearn.preprocessing import MinMaxScaler
from math import floor
from algebra import sep_contract_torch, contract_up
import os

module_dir = os.path.dirname(os.path.abspath(__file__))

############# DATASET HANDLING #############
############################################


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


# quantum embedding
def spin_map(tensor, dim=2):  # TODO: extend to higher dimensions
    cos = torch.cos(torch.pi * tensor / 2)
    sin = torch.sin(torch.pi * tensor / 2)

    assert torch.allclose(cos**2 + sin**2, torch.ones_like(cos))
    return torch.stack([cos, sin], dim=-1)


def poly_map(tensor, dim=2):

    powers = (
        torch.stack([tensor**i for i in range(dim)], dim=-1) + 1e-15
    )  # add a small number to avoid division by zero
    result = powers / torch.linalg.vector_norm(powers, dim=-1, keepdim=True)
    assert torch.allclose(
        torch.sum(result**2, dim=-1), torch.ones_like(tensor, dtype=result.dtype)
    )
    return result


mappings_dict = {"spin": spin_map, "poly": poly_map}


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
            tv.transforms.Lambda(partial(mappings_dict[mapping], dim=dim)),
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
    path=module_dir + "/../data",
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
    path=module_dir + "/../data",
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

    train = mappings_dict[mapping](load_to_device(train, device), dim=dim).to(
        dtype=dtype
    )
    test = mappings_dict[mapping](load_to_device(test, device), dim=dim).to(dtype=dtype)

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
    path=module_dir + "/../data",
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

    train = mappings_dict[mapping](
        linearize(load_to_device(train, device)), dim=dim
    ).to(dtype=dtype)
    test = mappings_dict[mapping](linearize(load_to_device(test, device)), dim=dim).to(
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
        h * w,
    )


def get_iris_data_loaders(
    batch_size,
    sel_labels=["Iris-setosa", "Iris-versicolor"],
    dtype=torch.double,
    mapping: str = "spin",
    dim=2,
    device="cpu",
    path=module_dir + "/../data",
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

    data = mappings_dict[mapping](load_to_device(data, device), dim=dim).to(dtype=dtype)
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
    path=module_dir + "/../data",
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
    data = mappings_dict[mapping](load_to_device(data, device), dim=dim).to(dtype=dtype)

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
    path=module_dir + "/../data",
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
    train_data = mappings_dict[mapping](load_to_device(train_data, device), dim=dim).to(
        dtype=dtype
    )
    test_data = mappings_dict[mapping](load_to_device(test_data, device), dim=dim).to(
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
    path=module_dir + "/../data",
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
    path=module_dir + "/../data",
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
    data = mappings_dict[mapping](load_to_device(data, device), dim=dim).to(dtype=dtype)

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


############# UTILS #############
#################################


def accuracy(
    model,
    device,
    train_dl,
    test_dl,
    dtype=torch.complex128,
    disable_pbar=False,
    quantize=False,
):
    correct = 0
    total = 0

    model.eval()
    model.to(device)

    with torch.no_grad():
        for data in tqdm(
            test_dl, total=len(test_dl), position=0, desc="test", disable=disable_pbar
        ):
            images, labels = data
            images, labels = images.to(device, dtype=dtype).squeeze(), labels.to(device)
            outputs = model(images, quantize=quantize)
            probs = torch.pow(torch.abs(outputs), 2)
            if model.n_labels > 1:
                _, predicted = torch.max(probs.data, 1)
                correct += (predicted == torch.where(labels == 1)[-1]).sum().item()
            else:
                predicted = torch.round(
                    probs.squeeze().data
                )  #! this is not correct, you should use a threshold!!!
                correct += (predicted == labels).sum().item()
            total += labels.size(0)

        test_accuracy = correct / total

        correct = 0
        total = 0

        for data in tqdm(
            train_dl,
            total=len(train_dl),
            position=0,
            desc="train",
            disable=disable_pbar,
        ):
            images, labels = data
            images, labels = images.to(device, dtype=dtype).squeeze(), labels.to(device)
            outputs = model(images)
            probs = torch.pow(torch.abs(outputs), 2)
            if model.n_labels > 1:
                _, predicted = torch.max(probs.data, 1)
                correct += (predicted == torch.where(labels == 1)[-1]).sum().item()
            else:
                predicted = torch.round(
                    probs.squeeze().data
                )  #! this is not correct, you should use a threshold!!!
                correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total

    return train_accuracy, test_accuracy


def accuracy_trenti(
    model,
    device,
    train_dl,
    test_dl,
    dtype=torch.complex128,
    disable_pbar=False,
    quantize=False,
):
    correct = 0
    total = 0

    model.eval()
    model.to(device)

    with torch.no_grad():
        for data in tqdm(
            test_dl, total=len(test_dl), position=0, desc="test", disable=disable_pbar
        ):
            images, labels = data
            images, labels = images.to(device, dtype=dtype).squeeze(), labels.to(device)
            outputs = model(images, quantize=quantize)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().cpu().item()

            total += labels.size(0)

        test_accuracy = correct / total

        correct = 0
        total = 0

        for data in tqdm(
            train_dl,
            total=len(train_dl),
            position=0,
            desc="train",
            disable=disable_pbar,
        ):
            images, labels = data
            images, labels = images.to(device, dtype=dtype).squeeze(), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().cpu().item()

            total += labels.size(0)

        train_accuracy = correct / total

    return train_accuracy, test_accuracy


def one_epoch_one_tensor(
    tensor, data_tn_batched, train_dl, optimizer, loss_fn, device="cuda", pbar=None
):
    # perform one epoch of optimization of a single tensor
    # given the data_tn and the optimizer
    tot_data = 0
    lossess = []
    n_labels = tensor.shape[-1]
    if pbar is None:
        pbar = tqdm(data_tn_batched, total=len(data_tn_batched), position=0)
    with torch.autograd.set_detect_anomaly(True):
        for data_tn, batch in zip(data_tn_batched, train_dl):
            optimizer.zero_grad()
            labels = batch[1].to(device=device)
            data = torch.stack([tensor.data for tensor in data_tn["l1"]])
            outputs = sep_contract_torch([tensor], data)

            probs = torch.pow(torch.abs(outputs), 2)

            if n_labels > 1:
                probs = probs / torch.sum(probs, -1, keepdim=True)
            loss = loss_fn(labels, probs, [tensor])

            loss.backward()
            optimizer.step()
            lossess.append(loss.cpu())
            tot_data += labels.shape[0]
            pbar.update()
            pbar.set_postfix_str(f"loss: {loss.item():.3f}")

    pbar.set_postfix(
        {
            "loss": loss.item(),
            "epoch mean loss": np.array([loss.item() for loss in lossess]).sum()
            / tot_data,
        }
    )
    pbar.close()
    return lossess


def one_epoch_one_tensor_torch(
    tensor,
    data_batched,
    train_dl,
    optimizer,
    loss_fn,
    device="cuda",
    pbar=None,
    disable_pbar=False,
):
    # perform one epoch of optimization of a single tensor
    # given the data_tn and the optimizer
    tot_data = 0
    lossess = []
    n_labels = tensor.shape[-1]
    if pbar is None:
        pbar = tqdm(
            data_batched, total=len(data_batched), position=0, disable=disable_pbar
        )
    with torch.autograd.set_detect_anomaly(True):
        for data, batch in zip(data_batched, train_dl):

            optimizer.zero_grad()
            labels = batch[1].to(device=device)

            outputs = contract_up(tensor, data.unbind(1))

            probs = torch.pow(torch.abs(outputs), 2)

            loss = loss_fn(labels, probs, [tensor])

            loss.backward()
            optimizer.step()
            lossess.append(loss.cpu())
            tot_data += labels.shape[0]
            pbar.update()
            pbar.set_postfix_str(f"loss: {loss.item():.3f}")

    pbar.set_postfix(
        {
            "loss": loss.item(),
            "epoch mean loss": np.array([loss.item() for loss in lossess]).mean(),
        }
    )
    pbar.close()
    return lossess


def train_one_epoch(
    model,
    device,
    train_dl,
    loss_fn,
    optimizer,
    quantize=False,
    gauging=False,
    pbar=None,
    disable_pbar=False,
):
    running_loss = 0.0
    last_loss = 0.0
    last_batch = 0
    loss_history = []
    close_pbar = False
    if pbar is None:
        close_pbar = True
        pbar = tqdm(
            enumerate(train_dl),
            total=len(train_dl),
            position=0,
            leave=True,
            disable=disable_pbar,
        )
    for i, data in enumerate(train_dl):
        if gauging:
            # set the center of the model to None as weights have been updated and the model is not canonical anymore
            model.center = None
            # normalize (the model will be gauged towards 0.0)
            model.canonicalize("0.0")

        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.squeeze().to(
            device, dtype=model.dtype, non_blocking=True
        ), labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs, quantize=quantize)
        probs = torch.pow(torch.abs(outputs), 2)

        # if model.n_labels > 1:
        #    probs = probs / torch.sum(probs, -1, keepdim=True, dtype=torch.float64)

        # Compute the loss and its gradients
        loss = loss_fn(labels, probs, [model.tensors[0]] if gauging else model.tensors)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        loss_history.append(loss.item())
        if i % 10 == 9:
            last_batch = i + 1
            last_loss = running_loss / 10  # mean loss over 10 batches
            running_loss = 0.0
        pbar.update()
        pbar.set_postfix(
            {
                "current loss": loss.item(),
                f"batches {last_batch-10}-{last_batch} loss": last_loss,
                "weight_norm": torch.as_tensor(
                    [torch.norm(tensor) for tensor in model.tensors]
                )
                .mean(0)
                .item(),
            }
        )

    pbar.set_postfix(
        {
            "current loss": loss.item(),
            f"batches {last_batch-10}-{last_batch} loss": last_loss,
            "epoch mean loss": np.array(loss_history).mean(),
        }
    )  # not correct as the last batch is averaged on less samples
    if close_pbar:
        pbar.close()
    if not gauging:
        model.center = None
    return loss_history


def class_loss_fn(labels, output: torch.Tensor, weights, l=0.1):
    loss_value = 0.0
    # regularization
    if l > 0.0:
        norms = torch.stack([torch.norm(tensor) for tensor in weights])
        target_norms = torch.sqrt(
            torch.tensor(
                [tensor.shape[-1] for tensor in weights],
                dtype=torch.float64,
                device=norms.device,
            )
        )

        loss_value += l * torch.mean((norms - target_norms) ** 2)

    # loss based on output dimension
    if output.squeeze().shape[-1] > 1:
        loss_value += torch.mean(torch.sum((output.squeeze() - labels) ** 2, -1)) / 2
    else:
        loss_value += torch.mean((output.squeeze() - labels) ** 2) / 2

    return loss_value


def dclass_loss_fn(output, labels):
    return (output - labels) / output.shape[0]


class ClassLoss(torch.nn.Module):
    def __init__(self, l=0.1, reduction="mean", transform=None):
        super(ClassLoss, self).__init__()
        self.l = l
        if l > 0.0:
            self.reg = torch.nn.MSELoss(reduction=reduction)
        self.bce = torch.nn.BCELoss(reduction=reduction)
        self.transform = transform

    def forward(self, labels: torch.Tensor, output: torch.Tensor, weights):
        loss_value = 0.0
        # regularization
        if self.l > 0.0:
            norms = torch.stack([torch.norm(tensor) for tensor in weights])
            # supposing we want all tensors to be isometries towards the up link
            target_norms = torch.sqrt(
                torch.tensor(
                    [tensor.shape[-1] for tensor in weights],
                    dtype=torch.float64,
                    device=norms.device,
                )
            )

            loss_value += self.l * self.reg(norms, target_norms)

        # bce
        if self.transform is not None:
            outputs = self.transform(output.squeeze())
        loss_value += self.bce(outputs, labels)

        return loss_value


def get_predictions(model, device, dl, disable_pbar=False):

    model.eval()
    model.to(device)
    predictions = []
    with torch.no_grad():
        for data in tqdm(
            dl, total=len(dl), position=0, desc="test", disable=disable_pbar
        ):
            images, labels = data
            images, labels = images.to(device, dtype=model.dtype).squeeze(), labels.to(
                device
            )
            outputs = model(images)
            probs = torch.pow(torch.abs(outputs), 2)
            # probs = probs / torch.sum(probs)
            predictions.append(probs.squeeze().detach().cpu())

    return torch.concat(predictions, dim=0)


def get_output(model, device, dl, disable_pbar=False, quantize=False):

    model.eval()
    model.to(device)
    outputs = []
    with torch.no_grad():
        for data in tqdm(
            dl, total=len(dl), position=0, desc="test", disable=disable_pbar
        ):
            images, labels = data
            images, labels = images.to(device, dtype=model.dtype).squeeze(), labels.to(
                device
            )
            outputs.append(model(images, quantize=quantize).squeeze().detach().cpu())

    return torch.concat(outputs, dim=0)


def search_label(do):
    for key, _ in do.items():
        if "label" in key.indices:
            return key

    return None


numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}

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
    return colorsys.hls_to_rgb(
        c_hls[0], max(0, min(1, amount * c_hls[1])), c_hls[2]
    ) + ((c[3],) if not rgb else ())
