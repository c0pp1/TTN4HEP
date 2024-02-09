import torch
import torchvision as tv
from quimb import tensor as qtn
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from sklearn.preprocessing import MinMaxScaler

from algebra import sep_contract_torch, contract_up

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
def spin_map(tensor, dim=2):                #TODO: extend to higher dimensions
    cos = torch.cos(torch.pi * tensor / 2)
    sin = torch.sin(torch.pi * tensor / 2)

    assert torch.allclose(cos**2 + sin**2, torch.ones_like(cos))
    return torch.stack([cos, sin], dim=-1)

def poly_map(tensor, dim=2):

    powers = torch.stack([tensor**i for i in range(dim)], dim=-1) + 1e-15 # add a small number to avoid division by zero
    result = powers / torch.linalg.vector_norm(powers, dim=-1, keepdim=True)
    assert(torch.allclose(torch.sum(result**2, dim=-1), torch.ones_like(tensor)))
    return result

mappings_dict = {
    'spin': spin_map,
    'poly': poly_map
}


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
    indices_balanced = np.stack([
        np.random.choice(train_index, max_class_count, replace=False)
        for train_index in indices
    ], axis=1)

    '''
    # get the balanced training and test sets
    balanced = torch.utils.data.Subset(
        samples, np.concatenate(indices_balanced)
    )
    '''
    return samples[indices_balanced.flatten()], labels[indices_balanced.flatten()]


def get_ttn_transform(h, mapping: str, dim=2, device="cpu"):
    return tv.transforms.Compose(
        [
            tv.transforms.Resize((h, h)),
            tv.transforms.ToTensor(),
            tv.transforms.Lambda(partial(load_to_device, device=device)),
            tv.transforms.Lambda(linearize),
            tv.transforms.Lambda(partial(mappings_dict[mapping], dim=dim)),
        ]
    )


def get_ttn_transform_visual(h):
    return tv.transforms.Compose(
        [tv.transforms.Resize((h, h)), tv.transforms.ToTensor()]
    )

def get_mnist_data_loaders(h, batch_size, labels=[0, 1], dtype=torch.double, mapping:str = 'spin', dim=2, device="cpu", path='../data'):
    # get the training and test sets
    train = tv.datasets.MNIST(
        root=path, train=True, download=True, transform=get_ttn_transform(h, mapping, dim, device)
    )
    test = tv.datasets.MNIST(
        root=path, train=False, download=True, transform=get_ttn_transform(h, mapping, dim, device)
    )
    train_visual = tv.datasets.MNIST(
        root=path, download=True, train=True, transform=get_ttn_transform_visual(h)
    )

    # balance the training and test sets
    train_balanced, train_labels = balance(train.targets, train.data, labels)
    test_balanced, test_labels = balance(test.targets, test.data, labels)

    train_balanced = torch.utils.data.TensorDataset(train_balanced.to(dtype=dtype), train_labels)
    test_balanced = torch.utils.data.TensorDataset(test_balanced.to(dtype=dtype), test_labels)

    NUM_WORKERS = torch.get_num_threads()

    test_dl = torch.utils.data.DataLoader(test_balanced, batch_size=batch_size)
    train_dl = torch.utils.data.DataLoader(train_balanced, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    train_visual = torch.utils.data.DataLoader(train_visual, batch_size=batch_size)

    return train_dl, test_dl, train_visual

def get_higgs_data_loaders(batch_size, dtype=torch.double, mapping:str = 'spin', dim=2, device="cpu", path='../data', permutation=None):
    # get the training and test sets
    train = torch.tensor(np.load(path + '/Higgs/higgs_train.npy'))
    test = torch.tensor(np.load(path + '/Higgs/higgs_test.npy'))
    train_labels = torch.tensor(np.load(path + '/Higgs/higgs_train_labels.npy'))
    test_labels = torch.tensor(np.load(path + '/Higgs/higgs_test_labels.npy'))

    if permutation is not None:
        train = train[:, permutation]
        test = test[:, permutation]

    train = mappings_dict[mapping](load_to_device(train, device), dim=dim).to(dtype=dtype)
    test = mappings_dict[mapping](load_to_device(test, device), dim=dim).to(dtype=dtype)

    train = torch.utils.data.TensorDataset(train, train_labels)
    test = torch.utils.data.TensorDataset(test, test_labels)

    NUM_WORKERS = torch.get_num_threads()
    return torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS), torch.utils.data.DataLoader(test, batch_size=batch_size), 8

def get_stripeimage_data_loaders(h, w, batch_size, dtype=torch.double, mapping:str = 'spin', dim=2, device="cpu", path='../data'):
    # get the training and test sets
    train = torch.tensor(np.load(path + f'/stripeimages/{h}x{w}train.npy'))
    test = torch.tensor(np.load(path + f'/stripeimages/{h}x{w}test.npy'))
    train_labels = torch.tensor(np.load(path + f'/stripeimages/{h}x{w}train_labels.npy'))
    test_labels = torch.tensor(np.load(path + f'/stripeimages/{h}x{w}test_labels.npy'))

    train = mappings_dict[mapping](linearize(load_to_device(train, device)), dim=dim).to(dtype=dtype)
    test = mappings_dict[mapping](linearize(load_to_device(test, device)), dim=dim).to(dtype=dtype)

    train = torch.utils.data.TensorDataset(train, train_labels)
    test = torch.utils.data.TensorDataset(test, test_labels)
    NUM_WORKERS = torch.get_num_threads()
    return torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS), torch.utils.data.DataLoader(test, batch_size=batch_size), h*w

def get_iris_data_loaders(batch_size, sel_labels=['Iris-setosa', 'Iris-versicolor'], dtype=torch.double, mapping:str = 'spin', dim=2, device="cpu", path='../data'):
    dataframe = pd.read_csv(path + '/iris/Iris.csv')
    dataframe = dataframe.sample(frac=1)


    dataframe['SepalLengthCm'] = dataframe['SepalLengthCm'] / dataframe['SepalLengthCm'].max()
    dataframe['SepalWidthCm'] = dataframe['SepalWidthCm'] / dataframe['SepalWidthCm'].max()
    dataframe['PetalLengthCm'] = dataframe['PetalLengthCm'] / dataframe['PetalLengthCm'].max()
    dataframe['PetalWidthCm'] = dataframe['PetalWidthCm'] / dataframe['PetalWidthCm'].max()

    dataframe = dataframe[dataframe['Species'].isin(sel_labels)]
    np.save(path + f'/iris/iris_index.npy', dataframe.index.to_numpy())
    dataframe = dataframe.reset_index(drop=True)
    labels = dataframe['Species'].to_numpy()
    labels = np.unique(labels, return_inverse=True)[1]
    np.save(path + f'/iris/iris_labels.npy', labels)
    labels = torch.nn.functional.one_hot(torch.tensor(labels), len(sel_labels))

    data = dataframe.drop(columns=['Id', 'Species']).to_numpy()
    np.save(path + f'/iris/iris_normalized.npy', data)
    data = torch.tensor(data)

    data = mappings_dict[mapping](load_to_device(data, device), dim=dim).to(dtype=dtype)
    train_size = int(0.8 * len(data))
    np.save(path + f'/iris/iris_embedded.npy', data)
    train = torch.utils.data.TensorDataset(data[:train_size], labels[:train_size])
    test = torch.utils.data.TensorDataset(data[train_size:], labels[train_size:])
    NUM_WORKERS = torch.get_num_threads()
    return torch.utils.data.DataLoader(train, batch_size=batch_size, num_workers=NUM_WORKERS), torch.utils.data.DataLoader(test, batch_size=batch_size), 4


def get_titanic_data_loaders(batch_size, dtype=torch.double, mapping:str = 'spin', dim=2, device="cpu", path='../data', scale=(0, 1)):
    dataframe = pd.read_csv(path + '/titanic/titanic.csv')
    #dataframe = dataframe.sample(frac=1)
    dataframe['sex'] = pd.Categorical(dataframe['sex']).codes
    dataframe['embarked'] = pd.Categorical(dataframe['embarked']).codes

    # transform not numerical ticket codes to numerical
    mask = np.array([not x.isdigit() for x in dataframe['ticket']])
    dataframe.loc[mask, 'ticket'] = pd.Categorical(dataframe['ticket'][mask]).codes
    dataframe['ticket'] = dataframe['ticket'].astype(int)

    labels = torch.tensor(dataframe['survived'].to_numpy())

    # scale the data
    scaler = MinMaxScaler(scale)
    df_scaled = pd.DataFrame(scaler.fit_transform(dataframe.drop(columns=['survived'])))

    # embed
    data = torch.tensor(df_scaled.to_numpy())
    data = mappings_dict[mapping](load_to_device(data, device), dim=dim).to(dtype=dtype)
    
    # balance the training and test sets
    data_balanced, labels_balanced = balance(labels, data)
    train_size = int(0.8 * len(data_balanced))

    train = torch.utils.data.TensorDataset(data_balanced[:train_size], labels_balanced[:train_size])
    test = torch.utils.data.TensorDataset(data_balanced[train_size:], labels_balanced[train_size:])

    NUM_WORKERS = torch.get_num_threads()
    return torch.utils.data.DataLoader(train, batch_size=batch_size, num_workers=NUM_WORKERS), \
           torch.utils.data.DataLoader(test, batch_size=batch_size), 8

############# UTILS #############
#################################

def accuracy(model, device, train_dl, test_dl, dtype=torch.complex128, disable_pbar=False):
    correct = 0
    total = 0

    model.eval()
    model.to(device)

    with torch.no_grad():
        for data in tqdm(test_dl, total=len(test_dl), position=0, desc='test', disable=disable_pbar):
            images, labels = data
            images, labels = images.to(device, dtype=dtype).squeeze(), labels.to(device)
            outputs = model(images)
            probs = torch.pow(torch.abs(outputs), 2)
            if model.n_labels > 1:
                _, predicted = torch.max(probs.data, 1)
                correct += (predicted == torch.where(labels == 1)[-1]).sum().item()
            else:
                predicted = torch.round(probs.squeeze().data)
                correct += (predicted == labels).sum().item()
            total += labels.size(0)
            

        test_accuracy = correct / total

        correct = 0
        total = 0

        for data in tqdm(train_dl, total=len(train_dl), position=0, desc='train', disable=disable_pbar):
            images, labels = data
            images, labels = images.to(device, dtype=dtype).squeeze(), labels.to(device)
            outputs = model(images)
            probs = torch.pow(torch.abs(outputs), 2)
            if model.n_labels > 1:
                _, predicted = torch.max(probs.data, 1)
                correct += (predicted == torch.where(labels == 1)[-1]).sum().item()
            else:
                predicted = torch.round(probs.squeeze().data)
                correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
        
        train_accuracy = correct / total

    return train_accuracy, test_accuracy


def one_epoch_one_tensor(tensor, data_tn_batched, train_dl, optimizer, loss_fn, device='cuda', pbar=None):
    # perform one epoch of optimization of a single tensor
    # given the data_tn and the optimizer
    tot_data = 0
    lossess = []
    n_labels = tensor.shape[-1]
    if pbar is None:
        pbar = tqdm(data_tn_batched, total=len(data_tn_batched),position=0)
    with torch.autograd.set_detect_anomaly(True):
        for data_tn, batch in zip(data_tn_batched, train_dl):
            optimizer.zero_grad()
            labels = batch[1].to(device=device)
            data = torch.stack([tensor.data for tensor in data_tn['l1']])
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
            pbar.set_postfix_str(f'loss: {loss.item():.3f}')
    
    pbar.set_postfix({'loss': loss.item(), 'epoch mean loss': np.array([loss.item() for loss in lossess]).sum() / tot_data})
    pbar.close()
    return lossess

def one_epoch_one_tensor_torch(tensor, data_batched, train_dl, optimizer, loss_fn, device='cuda', pbar=None, disable_pbar=False):
    # perform one epoch of optimization of a single tensor
    # given the data_tn and the optimizer
    tot_data = 0
    lossess = []
    n_labels = tensor.shape[-1]
    if pbar is None:
        pbar = tqdm(data_batched, total=len(data_batched),position=0, disable=disable_pbar)
    with torch.autograd.set_detect_anomaly(True):
        for data, batch in zip(data_batched, train_dl):
            optimizer.zero_grad()
            labels = batch[1].to(device=device)
            
            outputs = contract_up(tensor, data.unbind(1))

            probs = torch.pow(torch.abs(outputs), 2)

            if n_labels > 1:
                probs = probs / torch.sum(probs, -1, keepdim=True)
            loss = loss_fn(labels, probs, [tensor])

            loss.backward()
            optimizer.step()
            lossess.append(loss.cpu())
            tot_data += labels.shape[0]
            pbar.update()
            pbar.set_postfix_str(f'loss: {loss.item():.3f}')
    
    pbar.set_postfix({'loss': loss.item(), 'epoch mean loss': np.array([loss.item() for loss in lossess]).mean()})
    pbar.close()
    return lossess


def train_one_epoch(model, device, train_dl, loss_fn, optimizer, pbar=None, disable_pbar=False):
    running_loss = 0.
    last_loss = 0.
    last_batch = 0
    loss_history = []
    close_pbar = False
    if pbar is None:
        close_pbar = True
        pbar = tqdm(enumerate(train_dl), total=len(train_dl),position=0, leave=True, disable=disable_pbar)
    for i, data in enumerate(train_dl):

        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.squeeze().to(device, dtype=model.dtype, non_blocking=True), labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        probs = torch.pow(torch.abs(outputs), 2)

        if model.n_labels > 1:
            probs = probs / torch.sum(probs, -1, keepdim=True, dtype=torch.float64)
        
        # Compute the loss and its gradients
        loss = loss_fn(labels, probs, model.tensors)
        loss.backward()
        
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        loss_history.append(loss.item())
        if i % 10 == 9:
            last_batch = i+1
            last_loss = running_loss / 10 # mean loss over 10 batches
            running_loss = 0.  
        pbar.update()
        pbar.set_postfix({'current loss': loss.item(), 
                          f'batches {last_batch-10}-{last_batch} loss': last_loss, 
                          'weight_norm': torch.as_tensor([torch.norm(tensor) for tensor in model.tensors]).mean(0).item()})
        
    pbar.set_postfix({'current loss': loss.item(), f'batches {last_batch-10}-{last_batch} loss': last_loss, 'epoch mean loss': np.array(loss_history).mean()}) # not correct as the last batch is averaged on less samples
    if close_pbar:
        pbar.close()
    return loss_history


def class_loss(labels, output: torch.Tensor, weights, l=0.1):
    loss_value = 0.
    # regularization
    if l > 0.:
        norm = 0
        for tensor in weights:
            norm += torch.norm(tensor)
        norm /= len(weights)
        loss_value += l*(norm-1.)**2

    # loss based on output dimension
    if len(output.squeeze().shape) > 1:
        loss_value += torch.mean(torch.sum((output.squeeze() - labels)**2, -1))/2 
    else:
        loss_value += torch.mean((output.squeeze() - labels)**2)/2
    
    return loss_value


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