from __future__ import annotations
import numpy as np
from typing import Callable, Sequence
import torch
from tqdm.auto import tqdm

from ttnml.tn.ttn import TTNIndex, TTN
from ttnml.tn.tindex import TIndex
from ttnml.utils.miscellaneous import search_label

__all__ = ["TTNModel"]


# class which represents a Tree Tensor Network
# as a torch.nn.Module, so that it can be used
# as a model in a torch training loop.
class TTNModel(torch.nn.Module, TTN):
    def __init__(
        self,
        n_features,
        n_phys=2,
        n_labels=2,
        label_tag="label",
        bond_dim=8,
        dtype=torch.cdouble,
        device="cpu",
        quantizer=None,
    ):
        torch.nn.Module.__init__(self)
        TTN.__init__(
            self,
            n_features,
            n_phys,
            n_labels,
            label_tag,
            bond_dim,
            dtype,
            device,
            quantizer,
        )
        self.model_init = False

    @staticmethod
    def from_ttn(ttn: TTN, device="cpu", quantizer=None) -> TTNModel:
        ttn_model = TTNModel(
            ttn.n_features,
            ttn.n_phys,
            ttn.n_labels,
            ttn.label_tag,
            ttn.bond_dim,
            ttn.dtype,
            device,
            quantizer,
        )
        ttn_model.__dict__.update(ttn.__dict__)
        return ttn_model

    @staticmethod
    def from_npz(file_path: str, device="cpu", dtype=None, quantizer=None) -> TTNModel:
        ttn = TTN.from_npz(file_path, device=device, dtype=dtype, quantizer=quantizer)
        return TTNModel.from_ttn(ttn, device, quantizer)

    def forward(self, x: torch.Tensor, quantize=False):
        if not self.model_init:
            raise RuntimeError("TTNModel not initialized")
        data_dict = {
            TIndex(f"data.{i}", [f"data.{i}"]): datum
            for i, datum in enumerate(x.unbind(1))
        }

        result = self._propagate_data_through_branch_(
            data_dict, self.get_branch("0.0"), keep=True, quantize=quantize
        )["0.0"]

        return result * self.norm if self.normalized else result

    def initialize(
        self,
        dm_init=False,
        train_dl: torch.utils.data.Dataloader = None,
        loss_fn=None,
        epochs=5,
        disable_pbar=False,
        **kwargs,
    ):
        if dm_init:
            if (train_dl is None) or (loss_fn is None):
                raise ValueError(
                    f"The unsupervised and supervised initialization were invoked but the dataloader was: {train_dl}\n and the loss function was: {loss_fn}"
                )
            else:
                TTN.initialize(
                    self, train_dl, loss_fn, epochs, disable_pbar=disable_pbar, **kwargs
                )

        super(TTNModel, type(self)).tensors.fset(
            self,
            torch.nn.ParameterList(
                [torch.nn.Parameter(t, requires_grad=True) for t in self.tensors]
            ),
        )
        self.model_init = True

    def draw(self, **kwargs):
        return TTN.draw(self, **kwargs)

    def _set_gradient_(
        self,
        tindex: TTNIndex,
        out: torch.Tensor,
        data_mps: dict[TIndex, torch.Tensor],
        labels: torch.Tensor,
        dloss: Callable,
        loss: Callable,
        manual=True,
        return_grad=True,
    ):

        # get the derivative of the output with respect to the target tensor
        do = self.get_do_dt(tindex, data_mps)  # shape: (batch, bond_dim(, n_labels))
        if manual:
            with torch.no_grad():
                o2 = torch.pow(torch.abs(out), 2)
                if labels.dim() == 1:
                    dl = dloss(o2, labels.unsqueeze(1))
                else:
                    dl = dloss(o2, labels)
                dl *= 2 * out
        else:
            if loss is None:
                raise ValueError(
                    "A loss function must be passed in automatic gradient mode."
                )
            od = torch.nn.Parameter(out.detach(), requires_grad=True)
            od1 = torch.pow(torch.abs(od), 2)
            l = loss(labels, od1, [self[tindex]])
            l.backward()
            dl = od.grad
            if dl is None:
                raise ValueError("The gradient is None. This is not possible.")

        label_tensor = search_label(do)
        if label_tensor is not None:

            do[label_tensor] = torch.bmm(
                do[label_tensor], dl.unsqueeze(-1)
            ).squeeze()  # shape: (batch_dim, bond_dim)
        else:
            do[TIndex("label", ["label"])] = dl

        # get the gradient averaging over the batch dimension
        with torch.no_grad():
            grad = torch.einsum(
                "bi,bj,bk->bijk",
                do[tindex.indices[0]],
                do[tindex.indices[1]],
                do[tindex.indices[2]],
            ).mean(0)
        # set the gradient
        self._TTN__tensor_map[tindex].grad = grad

        if return_grad:
            return grad

    def sweep(
        self,
        data: Sequence[torch.Tensor] | torch.utils.data.dataloader.DataLoader,
        dloss: Callable,
        optimizer: torch.optim.Optimizer,
        epochs: int = 5,
        path_type: str = "layer",
        manual=True,
        loss=None,
        verbose=3,
        save_grads=True,
    ):
        path = []
        # data_mps = {TIndex(f'data.{i}', [f'data.{i}']): datum for i, datum in enumerate(data[0].unbind(1))}
        # labels = data[1]

        match path_type:
            case "layer":
                path = [
                    tindex
                    for i in range(self.n_layers)
                    for tindex in self.get_layer(i).keys()
                ]
            case "layer+0":
                path = list(self.get_layer(0).keys()) + [
                    tindex
                    for i in range(1, self.n_layers)
                    for tindex in (
                        list(self.get_layer(i).keys()) + list(self.get_layer(0).keys())
                    )
                ]
            case _:
                raise ValueError(f"The path type not implemented. Got {path_type}.")

        pbar_sweep = tqdm(
            path, desc="ttn sweep", position=0, leave=True, disable=verbose < 1
        )
        pbar_epoch = tqdm(
            total=epochs, desc="epochs", position=1, leave=True, disable=verbose < 2
        )
        if isinstance(data, torch.utils.data.dataloader.DataLoader):
            pbar_batch = tqdm(
                data,
                total=len(data),
                desc="batch",
                position=2,
                leave=True,
                disable=verbose < 3,
            )
        losses = []
        grads = []
        for tindex in pbar_sweep:

            pbar_sweep.set_description_str(f"ttn swwep, tensor {tindex.name}")
            # isometrise
            self.canonicalize(tindex)
            pbar_epoch.reset()
            for epoch in range(epochs):

                if isinstance(data, torch.utils.data.dataloader.DataLoader):
                    pbar_batch.reset()
                    batches_loss = []
                    for data_batch, labels in data:
                        # put gradients to zero
                        optimizer.zero_grad()
                        data_batch = data_batch.to(self.device, dtype=self.dtype)
                        labels = labels.to(self.device, dtype=self.dtype)

                        data_dict = {
                            TIndex(f"data.{i}", [f"data.{i}"]): datum
                            for i, datum in enumerate(data_batch.unbind(1))
                        }
                        with torch.no_grad():
                            out = self._propagate_data_through_branch_(
                                data_dict, self.get_branch("0.0"), keep=True
                            )["0.0"]
                            if loss is not None:
                                l = loss(
                                    labels,
                                    torch.abs(out.squeeze()) ** 2,
                                    [self[tindex]],
                                )
                                batches_loss.append(l.item())
                                pbar_epoch.set_postfix_str(
                                    f"loss: {np.array(batches_loss)[-10:].mean()}"
                                )

                        grad = self._set_gradient_(
                            tindex,
                            out,
                            data_dict,
                            labels,
                            dloss,
                            loss,
                            manual=manual,
                            return_grad=save_grads,
                        )
                        if save_grads:
                            grads.extend(grad.flatten().detach().cpu().numpy())
                        optimizer.step()
                        pbar_batch.update(1)
                    losses.append(np.array(batches_loss).mean())
                    pbar_batch.refresh()
                else:
                    optimizer.zero_grad()
                    data_batch, labels = data
                    data_batch = data_batch.to(self.device, dtype=self.dtype)
                    labels = labels.to(self.device, dtype=self.dtype)

                    data_dict = {
                        TIndex(f"data.{i}", [f"data.{i}"]): datum
                        for i, datum in enumerate(data_batch.unbind(1))
                    }
                    # propagate data through theoptimizer.zero_grad() branch without tracking gradients
                    with torch.no_grad():
                        out = self._propagate_data_through_branch_(
                            data_dict, self.get_branch("0.0"), keep=True
                        )["0.0"]
                        if loss is not None:
                            l = loss(
                                labels, torch.abs(out.squeeze()) ** 2, [self[tindex]]
                            )

                            pbar_epoch.set_postfix_str(f"loss: {l.item()}")
                            losses.append(l.item())

                    grad = self._set_gradient_(
                        tindex,
                        out,
                        data_dict,
                        labels,
                        dloss,
                        loss,
                        manual=manual,
                        return_grad=save_grads,
                    )
                    if save_grads:
                        grads.extend(grad.flatten().detach().cpu().numpy())
                    optimizer.step()

                # update the tensor

                pbar_epoch.update(1)
            pbar_epoch.refresh()
            pbar_sweep.set_postfix_str(f"loss: {np.array(losses).mean()}")
        pbar_epoch.close()
        pbar_sweep.close()
        if isinstance(data, torch.utils.data.dataloader.DataLoader):
            pbar_batch.close()

        self._TTN__norm = None
        self._TTN__normalized = False

        return losses, grads
