import numpy as np


__all__ = [
    "get_lipchitz_constant",
   "get_lipschitz_criterion",
    "is_smooth",
    "get_smoothness_threshold",
    "is_norm_bounded",
    "get_norm_bound_threshold"

]

def get_lipchitz_constant(X):
    """
    Compute the Lipschitz constant of the data.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data.

    Returns
    -------
    lipchitz_constant : float
        The Lipschitz constant of the data.
    """
    X = np.array(X)
    return np.max(np.abs(X - np.roll(X, 1, axis=0)))

def get_lipschitz_criterion(X):
    """
    Compute the Lipschitz criterion of the data.
    for proving the continuity of the data.
    (it is continuous if the derivative is constant)?
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data.

    Returns
    -------
    lipschitz_criterion : float
        The Lipschitz criterion of the data.
    """
    X = np.array(X)
    i = 0
    j = len(X)-1
    return np.max(abs(X[i] - X[j])/abs(i-j))

def is_smooth(X, threshold=1):
    """
    Check if the data is smooth.
    (it is smooth if the derivatives are less than threshold)
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data.

    Returns
    -------
    is_smooth : bool
        True if the data is smooth, False otherwise.
    """
    X = np.array(X)
    derivatives = np.diff(X)
    print("Derivatives:", derivatives)
    if np.all(np.abs(derivatives) <= threshold):
        return True
    else:
        return False

def get_smoothness_threshold(X):
    """
    Compute the smoothness threshold of the data.
    (it is smooth if the derivatives are less than threshold)
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data.

    Returns
    -------
    smoothness_criterion : float
        The smoothness criterion of the data.
    """
    X = np.array(X)
    derivatives = np.diff(X)
    return np.max(np.abs(derivatives))

def get_norm_bound_threshold(X):
    """
    get the norm bound threshold
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data.

    Returns
    -------
    norm_bound : float
        The norm bound of the data.
    """
    return np.max(np.linalg.norm(X, axis=1))
def is_norm_bounded(X, bound):
    """
    Check if the data is norm bounded.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data.

    Returns
    -------
    is_norm_bounded : bool
        True if the data is norm bounded, False otherwise.
    """
    X = np.array(X)
    norm = np.linalg.norm(X)
    print("Norm:", norm)
    if norm <= bound:
        return True
    else:
        return False