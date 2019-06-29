import re


def natural_sort(s):
    """ Sort the given list in the way that humans expect."""
    # Reference: https://nedbatchelder.com/blog/200712/human_sorting.html
    def alphanum_key(s):
        return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)]
    s.sort(key=alphanum_key)
    return s
