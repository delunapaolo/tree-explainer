import numpy as np


def iterative_mean(i_iter, current_mean, x):
    """Iteratively calculates mean using
    http://www.heikohoffmann.de/htmlthesis/node134.html. Originally implemented
    in treeexplainer https://github.com/andosa/treeexplainer/pull/24

    :param i_iter: [int > 0] Current iteration.
    :param current_mean: [ndarray] Current value of mean.
    :param x: [ndarray] New value to be added to mean.

    :return: [ndarray] Updated mean.
    """

    return current_mean + ((x - current_mean) / (i_iter + 1))


def divide0(a, b, replace_with):
    """Divide two numbers but replace its result if division is not possible,
    e.g., when dividing a number by 0. No type-checking or agreement between
    dimensions is performed. Be careful!

    :param a: [ndarray or int or float] Numerator.
    :param b: [ndarray or int or float] Denominator.
    :param replace_with: [int or float] Return this number if a/b is not defined.

    :return: [ndarray or int or float] Result of division, cast by numpy to the
        best data type to hold it.
    """

    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        if isinstance(c, np.ndarray):
            c[np.logical_not(np.isfinite(c))] = replace_with
        else:
            if not np.isfinite(c):
                c = replace_with

    return c
