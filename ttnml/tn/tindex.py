from __future__ import annotations
from typing import Sequence, List
import numpy as np

__all__ = ["TIndex"]

########## CLASS FOR TENSOR INDEXING ##########
#################################################


# class which represent a generic tensor index,
# it is used to identify a tensor in the TTN.
# It is composed by a name and a list of indices
class TIndex:
    """
    Index class for tensors in the TTN. It is composed by a name and a list of indices,
    one for each leg of the tensor. Implements comparison operators to compare the TIndex
    with other indices or strings, and has hash and string representation methods.

    ...
    Parameters
    ----------
    name : str
        The name of the index.
    inds : Sequence[str] | np.ndarray
        The list of indices of the index.

    Attributes
    ----------
    name : str
        The name of the index.
    indices : np.ndarray
        The list of indices in the TIndex.
    ndims : int
        The number of dimensions of the TIndex, i.e. the order of the tensor.

    """

    def __init__(self, name: str, inds: Sequence[str] | np.ndarray):
        self.__name = name
        self.__tindices = np.array(inds, dtype=np.str_)
        self.__ndims = len(inds)

    def __getitem__(self, key: int) -> str:
        return self.__tindices[key]

    def __setitem__(self, key: int, value: str):
        old_len = self.__tindices.dtype.itemsize / 4
        new_len = max(old_len, len(value))
        self.__tindices = self.__tindices.astype(f"<U{new_len:.0f}")
        self.__tindices[key] = value

    @property
    def name(self):
        return self.__name

    @property
    def indices(self):
        return self.__tindices

    @property
    def ndims(self):
        return self.__ndims

    """ I do not want them to be changed by design   
    @indices.setter
    def indices(self, value: Sequence[str]):
        self.__indices = value
    """

    def __eq__(self, __value: TIndex | str) -> bool:
        if isinstance(__value, str):
            return self.__name == __value
        return (
            self.__name == __value.name
            and np.all(self.__tindices == __value.indices).item()
        )

    def __gt__(self, __value: TIndex | str) -> bool:
        compare = __value if isinstance(__value, str) else __value.name
        try:
            compare_layer = int(compare.split(".")[0])
        except:
            compare_layer = np.inf
        try:
            self_layer = int(self.__name.split(".")[0])
        except:
            self_layer = np.inf

        if self_layer > compare_layer:
            return True
        elif self_layer == compare_layer:
            return int(self.__name.split(".")[1]) > int(compare.split(".")[1])
        return False

    def __lt__(self, __value: TIndex | str) -> bool:
        compare = __value if isinstance(__value, str) else __value.name
        try:
            compare_layer = int(compare.split(".")[0])
        except:
            compare_layer = np.inf
        try:
            self_layer = int(self.__name.split(".")[0])
        except:
            self_layer = np.inf

        if self_layer < compare_layer:
            return True
        elif self_layer == compare_layer:
            return int(self.__name.split(".")[1]) < int(compare.split(".")[1])
        return False

    def __ge__(self, __value: TIndex | str) -> bool:
        return self.__gt__(__value) or self.__eq__(__value)

    def __le__(self, __value: TIndex | str) -> bool:
        return self.__lt__(__value) or self.__eq__(__value)

    def __hash__(self):
        return hash(self.__name)

    def __str__(self) -> str:
        return self.__name

    def __repr__(self) -> str:
        return "TIndex: " + self.__name

    def _repr_markdown_(self):
        return f"**{self.__repr__()}**"

    def _repr_html_(self):
        markdown_str = f'<details><summary><b style="color:#0088d9; font-size:100%; font-family: verdana, sans-serif">{self.__repr__()} </b></summary>'
        for index in self.__tindices:
            markdown_str += f'&emsp;&ensp; <b style="color:#be00d9">{index}</b><br>'
        return markdown_str + "</details>"


# MEMO: THIS CLASS IS NOT USED
# class which represent the link between two tensors in the TTN
# it is composed by a source and a target tensor index, a dimension,
# a dependencies list and a name
class TLink:
    def __init__(
        self,
        source: TIndex,
        target: TIndex,
        dim: int,
        dependencies: List[TIndex] = [],
        name: str | None = None,
    ):
        self.__source = source
        self.__target = target
        self.__dim = dim
        self.__vector = None
        self.__has_updated_vector = False
        self.__name = name if name is not None else f"{source.name}"
        self.__dependencies = dependencies

    @property
    def source(self):
        return self.__source

    @property
    def target(self):
        return self.__target

    @property
    def name(self):
        return self.__name

    @property
    def vector(self):
        if self.__has_updated_vector:
            return self.__vector
        elif self.__vector is not None:
            raise ValueError(f"No vector is set for TLink {self.__name}.")
        else:
            raise ValueError(f"TLink {self.__name} vector is not updated.")

    @vector.setter
    def vector(self, value):
        self.__vector = value
        self.__has_updated_vector = True

    @property
    def is_updated(self):
        return self.__has_updated_vector

    def depends_on(self, __value: TIndex | str) -> bool:
        return __value in self.__dependencies
