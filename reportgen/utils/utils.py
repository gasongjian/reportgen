"""Small plotting-related utility functions."""
from __future__ import print_function, division
import colorsys
import warnings
import os

import numpy as np
from scipy import stats
import pandas as pd





def _kde_support(data, bw, gridsize, cut, clip):
    """Establish support for a kernel density estimate."""
    support_min = max(data.min() - bw * cut, clip[0])
    support_max = min(data.max() + bw * cut, clip[1])
    return np.linspace(support_min, support_max, gridsize)


def percentiles(a, pcts, axis=None):
    """Like scoreatpercentile but can take and return array of percentiles.

    Parameters
    ----------
    a : array
        data
    pcts : sequence of percentile values
        percentile or percentiles to find score at
    axis : int or None
        if not None, computes scores over this axis

    Returns
    -------
    scores: array
        array of scores at requested percentiles
        first dimension is length of object passed to ``pcts``

    """
    scores = []
    try:
        n = len(pcts)
    except TypeError:
        pcts = [pcts]
        n = 0
    for i, p in enumerate(pcts):
        if axis is None:
            score = stats.scoreatpercentile(a.ravel(), p)
        else:
            score = np.apply_along_axis(stats.scoreatpercentile, axis, a, p)
        scores.append(score)
    scores = np.asarray(scores)
    if not n:
        scores = scores.squeeze()
    return scores


def ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return percentiles(a, p, axis)


def iqr(a):
    """Calculate the IQR for an array of numbers."""
    a = np.asarray(a)
    q1 = stats.scoreatpercentile(a, 25)
    q3 = stats.scoreatpercentile(a, 75)
    return q3 - q1



def categorical_order(values, order=None):
    """Return a list of unique data values.

    Determine an ordered list of levels in ``values``.

    Parameters
    ----------
    values : list, array, Categorical, or Series
        Vector of "categorical" values
    order : list-like, optional
        Desired order of category levels to override the order determined
        from the ``values`` object.

    Returns
    -------
    order : list
        Ordered list of category levels not including null values.

    """
    if order is None:
        if hasattr(values, "categories"):
            order = values.categories
        else:
            try:
                order = values.cat.categories
            except (TypeError, AttributeError):
                try:
                    order = values.unique()
                except AttributeError:
                    order = pd.unique(values)
                try:
                    np.asarray(values).astype(np.float)
                    order = np.sort(order)
                except (ValueError, TypeError):
                    order = order
        order = filter(pd.notnull, order)
    return list(order)



