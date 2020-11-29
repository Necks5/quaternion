# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

__version__ = "2020.11.2.17.0.49"
__doc_title__ = "Quaternion dtype for NumPy"
__doc__ = "Adds a quaternion dtype to NumPy."
__all__ = ['dual',
           'as_float_array', 'from_float_array',
           'allclose',
           'zero', 'one', 'x',]


import numpy as np

from ._dual_number import (
    dual_number,
)

np.dual_number = dual_number
np.typeDict['dual_number'] = np.dtype(dual_number)

zero = np.dual_number(0, 0)
one = np.dual_number(1, 0)
x = np.dual_number(0, 1)


def as_float_array(a):
    """View the quaternion array as an array of floats

    This function is fast (of order 1 microsecond) because no data is
    copied; the returned quantity is just a "view" of the original.

    The output view has one more dimension (of size 4) than the input
    array, but is otherwise the same shape.

    """
    return np.asarray(a, dtype=np.dual_number).view((np.double, 2))


def as_quat_array(a):
    """View a float array as an array of quaternions

    The input array must have a final dimension whose size is
    divisible by four (or better yet *is* 4), because successive
    indices in that last dimension will be considered successive
    components of the output quaternion.

    This function is usually fast (of order 1 microsecond) because no
    data is copied; the returned quantity is just a "view" of the
    original.  However, if the input array is not C-contiguous
    (basically, as you increment the index into the last dimension of
    the array, you just move to the neighboring float in memory), the
    data will need to be copied which may be quite slow.  Therefore,
    you should try to ensure that the input array is in that order.
    Slices and transpositions will frequently break that rule.

    We will not convert back from a two-spinor array because there is
    no unique convention for them, so I don't want to mess with that.
    Also, we want to discourage users from the slow, memory-copying
    process of swapping columns required for useful definitions of
    the two-spinors.

    """
    a = np.asarray(a, dtype=np.double)

    # fast path
    if a.shape == (4,):
        return quaternion(a[0], a[1], a[2], a[3])

    # view only works if the last axis is C-contiguous
    if not a.flags['C_CONTIGUOUS'] or a.strides[-1] != a.itemsize:
        a = a.copy(order='C')
    try:
        av = a.view(np.dual_number)
    except ValueError as e:
        message = (str(e) + '\n            '
                   + 'Failed to view input data as a series of quaternions.  '
                   + 'Please ensure that the last dimension has size divisible by 4.\n            '
                   + 'Input data has shape {0} and dtype {1}.'.format(a.shape, a.dtype))
        raise ValueError(message)

    # special case: don't create an axis for a single quaternion, to
    # match the output of `as_float_array`
    if av.shape[-1] == 1:
        av = av.reshape(a.shape[:-1])

    return av


def from_float_array(a):
    return as_quat_array(a)


def isclose(a, b, rtol=4*np.finfo(float).eps, atol=0.0, equal_nan=False):
    """
    Returns a boolean array where two arrays are element-wise equal within a
    tolerance.

    This function is essentially a copy of the `numpy.isclose` function,
    with different default tolerances and one minor changes necessary to
    deal correctly with quaternions.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.

    Returns
    -------
    y : array_like
        Returns a boolean array of where `a` and `b` are equal within the
        given tolerance. If both `a` and `b` are scalars, returns a single
        boolean value.

    See Also
    --------
    allclose

    Notes
    -----
    For finite values, isclose uses the following equation to test whether
    two floating point values are equivalent:

      absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    The above equation is not symmetric in `a` and `b`, so that
    `isclose(a, b)` might be different from `isclose(b, a)` in
    some rare cases.

    Examples
    --------
    >>> quaternion.isclose([1e10*quaternion.x, 1e-7*quaternion.y], [1.00001e10*quaternion.x, 1e-8*quaternion.y],
    ...     rtol=1.e-5, atol=1.e-8)
    array([True, False])
    >>> quaternion.isclose([1e10*quaternion.x, 1e-8*quaternion.y], [1.00001e10*quaternion.x, 1e-9*quaternion.y],
    ...     rtol=1.e-5, atol=1.e-8)
    array([True, True])
    >>> quaternion.isclose([1e10*quaternion.x, 1e-8*quaternion.y], [1.0001e10*quaternion.x, 1e-9*quaternion.y],
    ...     rtol=1.e-5, atol=1.e-8)
    array([False, True])
    >>> quaternion.isclose([quaternion.x, np.nan*quaternion.y], [quaternion.x, np.nan*quaternion.y])
    array([True, False])
    >>> quaternion.isclose([quaternion.x, np.nan*quaternion.y], [quaternion.x, np.nan*quaternion.y], equal_nan=True)
    array([True, True])
    """
    def within_tol(x, y, atol, rtol):
        with np.errstate(invalid='ignore'):
            result = np.less_equal(abs(x-y), atol + rtol * abs(y))
        return result[()]

    x = np.array(a, copy=False, subok=True, ndmin=1)
    y = np.array(b, copy=False, subok=True, ndmin=1)

    # Make sure y is an inexact type to avoid bad behavior on abs(MIN_INT).
    # This will cause casting of x later. Also, make sure to allow subclasses
    # (e.g., for numpy.ma).
    try:
        dt = np.result_type(y, 1.)
    except TypeError:
        dt = np.dtype(np.dual_number)
    y = np.array(y, dtype=dt, copy=False, subok=True)

    xfin = np.isfinite(x)
    yfin = np.isfinite(y)
    if np.all(xfin) and np.all(yfin):
        return within_tol(x, y, atol, rtol)
    else:
        finite = xfin & yfin
        cond = np.zeros_like(finite, subok=True)
        # Because we're using boolean indexing, x & y must be the same shape.
        # Ideally, we'd just do x, y = broadcast_arrays(x, y). It's in
        # lib.stride_tricks, though, so we can't import it here.
        x = x * np.ones_like(cond)
        y = y * np.ones_like(cond)
        # Avoid subtraction with infinite/nan values...
        cond[finite] = within_tol(x[finite], y[finite], atol, rtol)
        # Check for equality of infinite values...
        cond[~finite] = (x[~finite] == y[~finite])
        if equal_nan:
            # Make NaN == NaN
            both_nan = np.isnan(x) & np.isnan(y)
            cond[both_nan] = both_nan[both_nan]

        return cond[()]


def allclose(a, b, rtol=4*np.finfo(float).eps, atol=0.0, equal_nan=False, verbose=False):
    """Returns True if two arrays are element-wise equal within a tolerance.

    This function is essentially a wrapper for the `quaternion.isclose`
    function, but returns a single boolean value of True if all elements
    of the output from `quaternion.isclose` are True, and False otherwise.
    This function also adds the option.

    Note that this function has stricter tolerances than the
    `numpy.allclose` function, as well as the additional `verbose` option.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.
    verbose : bool
        If the return value is False, all the non-close values are printed,
        iterating through the non-close indices in order, displaying the
        array values along with the index, with a separate line for each
        pair of values.

    See Also
    --------
    isclose, numpy.all, numpy.any, numpy.allclose

    Returns
    -------
    allclose : bool
        Returns True if the two arrays are equal within the given
        tolerance; False otherwise.


    Notes
    -----
    If the following equation is element-wise True, then allclose returns
    True.

      absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    The above equation is not symmetric in `a` and `b`, so that
    `allclose(a, b)` might be different from `allclose(b, a)` in
    some rare cases.

    """
    close = isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    result = np.all(close)
    if verbose and not result:
        a, b = np.atleast_1d(a), np.atleast_1d(b)
        a, b = np.broadcast_arrays(a, b)
        print('Non-close values:')
        for i in np.nonzero(close == False):
            print('    a[{0}]={1}\n    b[{0}]={2}'.format(i, a[i], b[i]))
    return result
