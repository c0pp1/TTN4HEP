import numpy as np
from typing import List, Sequence
import torch

__all__ = ["inv_permutation", "numpy_to_torch_dtype_dict", "search_label"]


# function to get the inverse permutation of a permutation
def inv_permutation(p: Sequence[int]) -> List[int]:
    return [p.index(i) for i in range(len(p))]


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
