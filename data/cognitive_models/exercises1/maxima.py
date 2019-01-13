# Advanced Scientific Programming in Python, autumn school, Trento, 2010
# Day 1, Exercise 1 (unit testing and coverage)
# Author: Pietro Berkes <berkes _at_ brandeis _dot_ edu>


def find_maxima(x):
    """Find local maxima of x.

    Example:
    >>> x = [1, 2, 3, 2, 4, 3]
    >>> find_maxima(x)
    [2, 4]

    Input arguments:
    x -- 1D list of real numbers

    Output:
    idx -- list of indices of the local maxima in x
    """

    idx = []
    for i in range(len(x)):
        # `i` is a local maximum if the signal decreases before and after it
        if x[i-1] < x[i] and x[i+1] < x[i]:
            idx.append(i)
    return idx

    # NOTE for the curious: the code above could be written using
    # list comprehension as
    # return [i for i in range(len(x)) if x[i-1]<x[i] and x[i+1]<x[i]]
    # not that this would solve the bugs ;-)
