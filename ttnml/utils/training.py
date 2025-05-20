import numpy as np
import torch
from tqdm import tqdm

from ttnml.tn.algebra import contract_up, sep_contract_torch

__all__ = [
    "accuracy",
    "accuracy_trenti",
    "one_epoch_one_tensor",
    "one_epoch_one_tensor_torch",
    "train_one_epoch",
    "class_loss_fn",
    "dclass_loss_fn",
    "ClassLoss",
    "get_predictions",
    "get_output",
]


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
    **kwargs,
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
            position=kwargs.get("position", 0),
            leave=kwargs.get("leave", True),
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
