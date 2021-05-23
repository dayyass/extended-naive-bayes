import numpy as np


def to_categorical(arr: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Perform One-Hot Encoding transformation on 1d numpy vector.

    :param np.ndarray arr: 1d numpy vector.
    :param int num_classes: number of classes.
    :return: 2d OHE matrix.
    :rtype: np.ndarray
    """

    assert arr.ndim == 1, "arr should be a 1d vector."

    ohe = np.zeros((len(arr), num_classes))
    ohe[np.arange(len(arr)), arr] = 1

    return ohe
