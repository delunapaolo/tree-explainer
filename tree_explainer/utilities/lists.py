import re
import numpy as np
import pandas as pd


def natural_sort(s):
    """ Sort the given list in the way that humans expect."""
    # Reference: https://nedbatchelder.com/blog/200712/human_sorting.html
    def alphanum_key(s):
        return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)]
    s.sort(key=alphanum_key)
    return s


def true_list(iterable):
    """Convert anything to a list."""
    if not isinstance(iterable, (list, np.ndarray, pd.Index)):
        result = list([iterable])

    else:
        if isinstance(iterable, list):
            result = list(iterable)
        else:
            result = iterable.tolist()

    return result
