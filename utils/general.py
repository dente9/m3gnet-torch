# m3gnet/utils/general.py

"""General utility functions."""

from typing import Optional, Sequence
import numpy as np

def check_array_equal(array1: Optional[np.ndarray], array2: Optional[np.ndarray], rtol: float = 1e-5) -> bool:
    """
    Check the equality of two optional numpy arrays.

    Args:
        array1 (np.ndarray | None): First array.
        array2 (np.ndarray | None): Second array.
        rtol (float): Relative tolerance for np.allclose.

    Returns:
        bool: True if arrays are equal or both are None.
    """
    if array1 is None and array2 is None:
        return True
    if array1 is None or array2 is None:
        return False
    return np.allclose(array1, array2, rtol=rtol)