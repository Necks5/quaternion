#!/usr/bin/env python

from __future__ import print_function, division, absolute_import
import os
import operator

import math
import numpy as np
from numpy.testing import assert_allclose
import pytest

from dual_number import dual
from dual_number import (allclose as d_allclose,
                         one as dual_one,
                         zero as dual_zero,
                         as_quat_array as dual_as_quat_array)

from sys import platform
on_windows = ('win' in platform.lower() and not 'darwin' in platform.lower())


eps = np.finfo(float).eps


def allclose(*args, **kwargs):
    kwargs.update({'verbose': True})
    return d_allclose(*args, **kwargs)


def passer(b):
    pass
# Change this to strict_assert = assert_ to check for missing tests
strict_assert = passer


def ufunc_binary_utility(array1, array2, op, rtol=2*eps, atol=0.0):
    """Make sure broadcasting is consistent with individual operations

    Given two arrays, we expect a broadcast binary operation to be consistent with the individual operations.  This
    utility function simply goes through and checks that that is true.  For example, if the input operation is `*`,
    this function checks for each `i` that

        array1[i] * array2  ==  np.array([array1[i]*array2[j] for j in range(len(array2))])

    """
    for arg1 in array1:
        assert allclose(op(arg1, array2),
                        np.array([op(arg1, arg2) for arg2 in array2]),
                        rtol=rtol, atol=atol)
    for arg2 in array2:
        assert allclose(op(array1, arg2),
                        np.array([op(arg1, arg2) for arg1 in array1]),
                        rtol=rtol, atol=atol)

    if array1.shape == array2.shape:
        assert allclose(op(array1, array2),
                        np.array([op(arg1, arg2) for arg1, arg2 in zip(array1, array2)]),
                        rtol=rtol, atol=atol)


# The following fixtures are used to establish some re-usable data
# for the tests; they need to be re-constructed because some of the
# tests will change the values, but we want the values to be constant
# on every entry into a test.

@pytest.fixture
def Qs():
    return make_Qs()
    
def make_Qs():
    q_nan1 = dual(np.nan, 0.)
    q_inf1 = dual(np.inf, 0.)
    q_minf1 = dual(-np.inf, 0.)
    q_0 = dual(0., 0.)
    q_1 = dual(1., 0.)
    x = dual(0., 1.)
    Q = dual(1.1, 2.2)
    Qneg = dual(-1.1, -2.2)
    Qbar = dual(1.1, -2.2)
    Qnormalized = dual(1.0, 2.0)
    Qlog = dual(1.7959088706354, 0.515190292664085,)
    Qexp = dual(2.81211398529184, -0.392521193481878,)
    return np.array([q_nan1, q_inf1, q_minf1, q_0, q_1, x, Q, Qneg, Qbar, Qnormalized, Qlog, Qexp],
                    dtype=np.dual)

Qs_array = make_Qs()


q_nan1, q_inf1, q_minf1, q_0, q_1, x, Q, Qneg, Qbar, Qnormalized, Qlog, Qexp, = range(len(Qs_array))
Qs_zero = [i for i in range(len(Qs_array)) if not Qs_array[i].nonzero()]
Qs_nonzero = [i for i in range(len(Qs_array)) if Qs_array[i].nonzero()]
Qs_nan = [i for i in range(len(Qs_array)) if Qs_array[i].isnan()]
Qs_nonnan = [i for i in range(len(Qs_array)) if not Qs_array[i].isnan()]
Qs_nonnannonzero = [i for i in range(len(Qs_array)) if not Qs_array[i].isnan() and Qs_array[i].nonzero()]
Qs_inf = [i for i in range(len(Qs_array)) if Qs_array[i].isinf()]
Qs_noninf = [i for i in range(len(Qs_array)) if not Qs_array[i].isinf()]
Qs_noninfnonzero = [i for i in range(len(Qs_array)) if not Qs_array[i].isinf() and Qs_array[i].nonzero()]
Qs_finite = [i for i in range(len(Qs_array)) if Qs_array[i].isfinite()]
Qs_nonfinite = [i for i in range(len(Qs_array)) if not Qs_array[i].isfinite()]
Qs_finitenonzero = [i for i in range(len(Qs_array)) if Qs_array[i].isfinite() and Qs_array[i].nonzero()]


@pytest.fixture
def Rs():
    ones = [0, -1., 1.]
    rs = [np.dual(w, x, y, z).normalized() for w in ones for x in ones for y in ones for z in ones][1:]
    np.random.seed(1842)
    rs = rs + [r.normalized() for r in [np.dual(np.random.uniform(-1, 1), np.random.uniform(-1, 1),
                                                      np.random.uniform(-1, 1), np.random.uniform(-1, 1)) for i in range(20)]]
    return np.array(rs)


def test_quaternion_members():
    Q = dual(1.1, 2.2)
    assert Q.real == 1.1
    assert Q.imag == 2.2


def test_quaternion_constructors():
    Q = dual(2.2, 3.3)
    assert Q.real == 2.2
    assert Q.imag == 3.3
    
    P = dual(1.1, 2.2)
    Q = dual(P)
    assert Q.real == 1.1
    assert Q.imag == 2.2

    Q = dual(1.1)
    assert Q.real == 1.1
    assert Q.imag == 0.0

    Q = dual()
    assert Q.real == 0.0
    assert Q.imag == 0.0

    with pytest.raises(TypeError):
        dual(1.2, 3.4, 4.5)

    with pytest.raises(TypeError):
        dual(1.2, 3.4, 5.6, 7.8, 9.0)

    with pytest.raises(TypeError):
        dual([1., 2.])


@pytest.mark.parametrize("q", make_Qs())
def test_bad_conversions(q):
    with pytest.raises(TypeError):
        s = int(q)
    with pytest.raises(TypeError):
        s = float(q)
    with pytest.raises(TypeError):
        a = np.zeros(3, dtype=int)
        a[0] = q
    with pytest.raises(TypeError):
        a = np.zeros(3)
        a[0] = q


# Unary bool returners
def test_quaternion_nonzero(Qs):
    assert not Qs[q_0].nonzero()  # Do this one explicitly, to not use circular logic
    assert Qs[q_1].nonzero()  # Do this one explicitly, to not use circular logic
    for q in Qs[Qs_zero]:
        assert not q.nonzero()
    for q in Qs[Qs_nonzero]:
        assert q.nonzero()


def test_quaternion_isnan(Qs):
    assert not Qs[q_0].isnan()  # Do this one explicitly, to not use circular logic
    assert not Qs[q_1].isnan()  # Do this one explicitly, to not use circular logic
    assert Qs[q_nan1].isnan()  # Do this one explicitly, to not use circular logic
    for q in Qs[Qs_nan]:
        assert q.isnan()
    for q in Qs[Qs_nonnan]:
        assert not q.isnan()


def test_quaternion_isinf(Qs):
    assert not Qs[q_0].isinf()  # Do this one explicitly, to not use circular logic
    assert not Qs[q_1].isinf()  # Do this one explicitly, to not use circular logic
    assert Qs[q_inf1].isinf()  # Do this one explicitly, to not use circular logic
    assert Qs[q_minf1].isinf()  # Do this one explicitly, to not use circular logic
    for q in Qs[Qs_inf]:
        assert q.isinf()
    for q in Qs[Qs_noninf]:
        assert not q.isinf()


def test_quaternion_isfinite(Qs):
    assert not Qs[q_nan1].isfinite()  # Do this one explicitly, to not use circular logic
    assert not Qs[q_inf1].isfinite()  # Do this one explicitly, to not use circular logic
    assert not Qs[q_minf1].isfinite()  # Do this one explicitly, to not use circular logic
    assert Qs[q_0].isfinite()  # Do this one explicitly, to not use circular logic
    for q in Qs[Qs_nonfinite]:
        assert not q.isfinite()
    for q in Qs[Qs_finite]:
        assert q.isfinite()


# Binary bool returners
def test_quaternion_equal(Qs):
    for j in Qs_nonnan:
        assert Qs[j] == Qs[j]  # self equality
        for k in range(len(Qs)):  # non-self inequality
            assert (j == k) or (not (Qs[j] == Qs[k]))
    for q in Qs:
        for p in Qs[Qs_nan]:
            assert not q == p  # nan should never equal anything


def test_quaternion_not_equal(Qs):
    for j in Qs_nonnan:
        assert not (Qs[j] != Qs[j])  # self non-not_equality
        for k in Qs_nonnan:  # non-self not_equality
            assert (j == k) or (Qs[j] != Qs[k])
    for q in Qs:
        for p in Qs[Qs_nan]:
            assert q != p  # nan should never equal anything


def test_quaternion_richcompare(Qs):
    for p in Qs:
        for q in Qs[Qs_nan]:
            assert not p < q
            assert not q < p
            assert not p <= q
            assert not q <= p
            assert not p.greater(q)
            assert not q.greater(p)
            assert not p.greater_equal(q)
            assert not q.greater_equal(p)
    for j in Qs_nonnan:
        p = Qs[j]
        assert (p < Qs[q_inf1]) or (j == q_inf1)
        assert (p <= Qs[q_inf1])
        assert (Qs[q_minf1] < p) or (j == q_minf1)
        assert (Qs[q_minf1] <= p)
        assert (Qs[q_inf1].greater(p)) or (j == q_inf1)
        assert (Qs[q_inf1].greater_equal(p))
        assert (p.greater(Qs[q_minf1])) or (j == q_minf1)
        assert (p.greater_equal(Qs[q_minf1]))
    for p in [Qs[q_1], Qs[x], Qs[Q], Qs[Qbar]]:
        assert Qs[q_0] < p
        assert Qs[q_0] <= p
        assert p.greater(Qs[q_0])
        assert p.greater_equal(Qs[q_0])
    for p in [Qs[Qneg]]:
        assert p < Qs[q_0]
        assert p <= Qs[q_0]
        assert Qs[q_0].greater(p)
        assert Qs[q_0].greater_equal(p)
    for p in [Qs[x]]:
        assert p < Qs[q_1]
        assert p <= Qs[q_1]
        assert Qs[q_1].greater(p)
        assert Qs[q_1].greater_equal(p)
    for p in [Qs[Qlog], Qs[Qexp]]:
        assert Qs[q_1] < p
        assert Qs[q_1] <= p
        assert p.greater(Qs[q_1])
        assert p.greater_equal(Qs[q_1])


# Unary float returners
def test_quaternion_absolute(Qs):
    for q in Qs[Qs_nan]:
        assert np.isnan(q.abs())
    for q in Qs[Qs_inf]:
        if on_windows:
            assert np.isinf(q.abs()) or np.isnan(q.abs())
        else:
            assert np.isinf(q.abs())
    
    for q, a in [(Qs[q_0], 0.0),
                 (Qs[q_1], 1.0),
                 (Qs[x], 0.0),
                 (Qs[Q], abs(Qs[Q].real)),
                 (Qs[Qbar], abs(Qs[Q].real)),
                 ]:
        assert np.allclose(q.abs(), a)


def test_quaternion_norm(Qs):
    for q in Qs[Qs_nan]:
        assert np.isnan(q.norm())
    for q in Qs[Qs_inf]:
        if on_windows:
            assert np.isinf(q.norm()) or np.isnan(q.norm())
        else:
            assert np.isinf(q.norm())

    for q, a in [(Qs[q_0], 0.0),
                 (Qs[q_1], 1.0),
                 (Qs[x], 0.0),
                 (Qs[Q], Qs[Q].real **2),
                 (Qs[Qbar], Qs[Q].real **2),
                 ]:
        assert np.allclose(q.norm(), a)


# Unary quaternion returners
def test_quaternion_negative(Qs):
    assert -Qs[Q] == Qs[Qneg]
    for q in Qs[Qs_finite]:
        assert -q == -1.0 * q
    for q in Qs[Qs_nonnan]:
        assert -(-q) == q


def test_quaternion_conjugate(Qs):
    assert Qs[Q].conjugate() == Qs[Qbar]
    for q in Qs[Qs_nonnan]:
        assert q.conjugate() == q.conj()
        assert q.conjugate().conjugate() == q
        c = q.conjugate()
        assert c.real == q.real
        assert c.imag == -q.imag


def test_quaternion_normalized(Qs):
    assert abs(Qs[Q].normalized()-Qs[Qnormalized]) < 4e-16
    for q in Qs[Qs_finitenonzero]:
        if q.real != 0:
            # abs not defined if q.real == 0, cf Yaglom
            assert abs(q.normalized().abs() - 1.0) < 1.e-15


# Quaternion-quaternion, scalar-quaternion, or quaternion-scalar binary quaternion returners
def test_quaternion_add(Qs):
    for j in Qs_nonnan:
        for k in Qs_nonnan:
            q = Qs[j]
            p = Qs[k]
            assert (q + p == dual(q.real + p.real, q.imag + p.imag)
                    or (j == q_inf1 and k == q_minf1)
                    or (k == q_inf1 and j == q_minf1))
    for q in Qs[Qs_nonnan]:
        for s in [-3, -2.3, -1.2, -1.0, 0.0, 0, 1.0, 1, 1.2, 2.3, 3]:
            assert (q + s == dual(q.real + s, q.imag))
            assert (s + q == dual(q.real + s, q.imag))


def test_quaternion_add_ufunc(Qs):
    ufunc_binary_utility(Qs[Qs_finite], Qs[Qs_finite], operator.add)


def test_quaternion_subtract(Qs):
    for q in Qs[Qs_finite]:
        for p in Qs[Qs_finite]:
            assert q - p == dual(q.real - p.real, q.imag - p.imag)
    for q in Qs[Qs_nonnan]:
        for s in [-3, -2.3, -1.2, -1.0, 0.0, 0, 1.0, 1, 1.2, 2.3, 3]:
            assert (q - s == dual(q.real - s, q.imag))
            assert (s - q == dual(s - q.real, -q.imag))


def test_quaternion_subtract_ufunc(Qs):
    ufunc_binary_utility(Qs[Qs_finite], Qs[Qs_finite], operator.sub)


def test_quaternion_multiply(Qs):
    # Check scalar multiplication
    for q in Qs[Qs_finite]:
        assert q * Qs[q_1] == q
    for q in Qs[Qs_finite]:
        assert q * 1.0 == q
        assert q * 1 == q
        assert 1.0 * q == q
        assert 1 * q == q
    for s in [-3, -2.3, -1.2, -1.0, 0.0, 0, 1.0, 1, 1.2, 2.3, 3]:
        for q in Qs[Qs_finite]:
            assert q * s == dual(s * q.real, s * q.imag)
            assert s * q == q * s
    for q in Qs[Qs_finite]:
        assert 0.0 * q == Qs[q_0]
        assert 0.0 * q == q * 0.0

    # Check linearity
    for q1 in Qs[Qs_finite]:
        for q2 in Qs[Qs_finite]:
            for q3 in Qs[Qs_finite]:
                assert allclose(q1*(q2+q3), (q1*q2)+(q1*q3))
                assert allclose((q1+q2)*q3, (q1*q3)+(q2*q3))

    # Check the multiplication table
    one = dual(1, 0)
    jay = dual(0, 1)
    assert one*one == one
    assert one*jay == jay
    assert jay*one == jay
    assert jay*jay == dual(0, 0)
    

def test_quaternion_multiply_ufunc(Qs):
    ufunc_binary_utility(np.array([dual_one]), Qs[Qs_finite], operator.mul)
    ufunc_binary_utility(Qs[Qs_finite], np.array([dual_one]), operator.mul)
    ufunc_binary_utility(np.array([1.0]), Qs[Qs_finite], operator.mul)
    
  
# XXX    ufunc_binary_utility(Qs[Qs_finite], np.array([1.0]), operator.mul)
    ufunc_binary_utility(np.array([1]), Qs[Qs_finite], operator.mul)
    ufunc_binary_utility(Qs[Qs_finite], np.array([1]), operator.mul)
    ufunc_binary_utility(np.array([0.0]), Qs[Qs_finite], operator.mul)
# XXX    ufunc_binary_utility(Qs[Qs_finite], np.array([0.0]), operator.mul)
    ufunc_binary_utility(np.array([0]), Qs[Qs_finite], operator.mul)
    ufunc_binary_utility(Qs[Qs_finite], np.array([0]), operator.mul)

    ufunc_binary_utility(np.array([-3, -2.3, -1.2, -1.0, 0.0, 0, 1.0, 1, 1.2, 2.3, 3]),
                         Qs[Qs_finite], operator.mul)
# XXX    ufunc_binary_utility(Qs[Qs_finite],
# XXX                         np.array([-3, -2.3, -1.2, -1.0, 0.0, 0, 1.0, 1, 1.2, 2.3, 3]), operator.mul)

    ufunc_binary_utility(Qs[Qs_finite], Qs[Qs_finite], operator.mul)


def test_quaternion_divide(Qs):
    # Check identity between "inverse" and "reciprocal"
    for q in Qs[Qs_finitenonzero]:
        if q.real != 0:
            assert q.inverse() == q.reciprocal()

    # Check scalar division
    for q in Qs[Qs_finitenonzero]:
        if q.real == 0:
            continue
        assert allclose(q / q, dual_one)
        assert allclose(1 / q, q.inverse())
        assert allclose(1.0 / q, q.inverse())
        assert 0.0 / q == dual_zero
        for s in [-3, -2.3, -1.2, -1.0, 0.0, 0, 1.0, 1, 1.2, 2.3, 3]:
            assert allclose(s / q, s * (q.inverse()))
            
    for q in Qs[Qs_finitenonzero]:
        assert q / 1.0 == q
        assert q / 1 == q
        for s in [-3, -2.3, -1.2, -1.0, 1.0, 1, 1.2, 2.3, 3]:
            assert allclose(q / s, q * (1.0/s))

    # Check linearity
    for q1 in Qs[Qs_finite]:
        for q2 in Qs[Qs_finite]:
            for q3 in Qs[Qs_finitenonzero]:
                if q3.real == 0:
                    continue
                assert allclose((q1+q2)/q3, (q1/q3)+(q2/q3))

    # Check the multiplication table
    one = dual(1, 0)
    jay = dual(0, 1)
    assert one / one == one
    assert jay / one == jay


def test_quaternion_divide_ufunc(Qs):

    # remove zero real parts
    qs_nonzero = Qs[Qs_finitenonzero]
    qs_nonzero = np.array([q for q in qs_nonzero if q.real != 0])

    one = dual(1., 0.)

    ufunc_binary_utility(np.array([one]), qs_nonzero, operator.truediv)
    ufunc_binary_utility(Qs[Qs_finite], np.array([one]), operator.truediv)
    ufunc_binary_utility(np.array([1.0]), qs_nonzero, operator.truediv)
    ufunc_binary_utility(Qs[Qs_finite], np.array([1.0]), operator.truediv)
    ufunc_binary_utility(np.array([1]), qs_nonzero, operator.truediv)
    ufunc_binary_utility(Qs[Qs_finite], np.array([1]), operator.truediv)
    ufunc_binary_utility(np.array([0.0]), qs_nonzero, operator.truediv)   # XXX: zero .real
    ufunc_binary_utility(np.array([0]), qs_nonzero, operator.truediv)

    ufunc_binary_utility(np.array([-3, -2.3, -1.2, -1.0, 0.0, 0, 1.0, 1, 1.2, 2.3, 3]),
                         qs_nonzero, operator.truediv)
    ufunc_binary_utility(Qs[Qs_finitenonzero],
                         np.array([-3, -2.3, -1.2, -1.0, 1.0, 1, 1.2, 2.3, 3]), operator.truediv)

    ufunc_binary_utility(Qs[Qs_finite], qs_nonzero, operator.truediv)
    ufunc_binary_utility(Qs[Qs_finite], qs_nonzero, operator.floordiv)

    ufunc_binary_utility(qs_nonzero, qs_nonzero, operator.truediv)
    ufunc_binary_utility(qs_nonzero, qs_nonzero, operator.floordiv)


def test_quaternion_getset(Qs):
    # get components/vec
    for q in Qs[Qs_nonnan]:
        assert np.array_equal(q.components, np.array([q.real, q.imag]))

    # set components/vec from np.array, list, tuple
    for q in Qs[Qs_nonnan]:
        for seq_type in [np.array, list, tuple]:
            p = np.dual(*q.components)
            r = np.dual(*q.components)
            s = np.dual(*q.components)
            p.components = seq_type((-5.5, 6.6))
            assert np.array_equal(p.components, np.array([-5.5, 6.6]))

    # TypeError when setting components with the wrong type or size of thing
    for q in Qs:
        for seq_type in [np.array, list, tuple]:
            p = np.dual(*q.components)
            with pytest.raises(TypeError):
                p.components = '1.1, 2.2'
            with pytest.raises(TypeError):
                p.components = seq_type([])
            with pytest.raises(TypeError):
                p.components = seq_type((-5.5,))
            with pytest.raises(TypeError):
                p.components = seq_type((-5.5, 6.6, -7.7,))
            with pytest.raises(TypeError):
                p.components = seq_type((-5.5, 6.6, -7.7, 8.8, -9.9))


def test_setitem_quat(Qs):
    Ps = Qs.copy()
    # setitem from quaternion
    for j in range(len(Ps)):
        Ps[j] = np.dual(1.3, 2.4)
        for k in range(j + 1):
            assert Ps[k] == np.dual(1.3, 2.4)
        for k in range(j + 1, len(Ps)):
            assert Ps[k] == Qs[k]
            
    # setitem from np.array, list, or tuple
    for seq_type in [np.array, list, tuple]:
        Ps = Qs.copy()
        with pytest.raises(TypeError):
            Ps[0] = seq_type(())
        with pytest.raises(TypeError):
            Ps[0] = seq_type((1.3,))
        with pytest.raises(TypeError):
            Ps[0] = seq_type((1.3, 2.4, 3.5))
        with pytest.raises(TypeError):
            Ps[0] = seq_type((1.3, 2.4, 3.5, 4.7, 5.9))
        with pytest.raises(TypeError):
            Ps[0] = seq_type((1.3, 2.4, 3.5, 4.7, 5.9, np.nan))
        for j in range(len(Ps)):
            Ps[j] = seq_type((1.3, 2.4))
            for k in range(j + 1):
                assert Ps[k] == np.dual(1.3, 2.4)
            for k in range(j + 1, len(Ps)):
                assert Ps[k] == Qs[k]
    with pytest.raises(TypeError):
        Ps[0] = 's'
    with pytest.raises(TypeError):
        Ps[0] = 's'

################################################# Seems to pass up to here


@pytest.mark.skip
def test_isclose():
    from quaternion import x, y

    assert np.array_equal(quaternion.isclose([1e10*x, 1e-7*y], [1.00001e10*x, 1e-8*y], rtol=1.e-5, atol=2.e-8),
                          np.array([True, False]))
    assert np.array_equal(quaternion.isclose([1e10*x, 1e-8*y], [1.00001e10*x, 1e-9*y], rtol=1.e-5, atol=2.e-8),
                          np.array([True, True]))
    assert np.array_equal(quaternion.isclose([1e10*x, 1e-8*y], [1.0001e10*x, 1e-9*y], rtol=1.e-5, atol=2.e-8),
                          np.array([False, True]))
    assert np.array_equal(quaternion.isclose([x, np.nan*y], [x, np.nan*y]),
                          np.array([True, False]))
    assert np.array_equal(quaternion.isclose([x, np.nan*y], [x, np.nan*y], equal_nan=True),
                          np.array([True, True]))

    np.random.seed(1234)
    a = quaternion.as_quat_array(np.random.random((3, 5, 4)))
    assert quaternion.allclose(1e10 * a, 1.00001e10 * a, rtol=1.e-5, atol=2.e-8, verbose=True) == True
    assert quaternion.allclose(1e-7 * a, 1e-8 * a, rtol=1.e-5, atol=2.e-8) == False
    assert quaternion.allclose(1e10 * a, 1.00001e10 * a, rtol=1.e-5, atol=2.e-8, verbose=True) == True
    assert quaternion.allclose(1e-8 * a, 1e-9 * a, rtol=1.e-5, atol=2.e-8, verbose=True) == True
    assert quaternion.allclose(1e10 * a, 1.0001e10 * a, rtol=1.e-5, atol=2.e-8) == False
    assert quaternion.allclose(1e-8 * a, 1e-9 * a, rtol=1.e-5, atol=2.e-8, verbose=True) == True
    assert quaternion.allclose(np.nan * a, np.nan * a) == False
    assert quaternion.allclose(np.nan * a, np.nan * a, equal_nan=True, verbose=True) == True



@pytest.mark.skip
def test_as_float_quat(Qs):
    qs = Qs[Qs_nonnan]
    for quats in [qs, np.vstack((qs,)*3), np.vstack((qs,)*(3*5)).reshape((3, 5)+qs.shape),
                  np.vstack((qs,)*(3*5*6)).reshape((3, 5, 6)+qs.shape)]:
        floats = quaternion.as_float_array(quats)
        assert floats.shape == quats.shape+(4,)
        assert allclose(quaternion.as_quat_array(floats), quats)
        assert allclose(quaternion.from_float_array(floats), quats)
        # Test that we can handle a list just like an array
        assert np.array_equal(quaternion.as_quat_array(floats), quaternion.as_quat_array(floats.tolist()))
    a = np.arange(12).reshape(3, 4)
    assert np.array_equal(quaternion.as_float_array(quaternion.as_quat_array(a)),
                          a.astype(float))
    assert quaternion.as_float_array(quaternion.x).ndim == 1



@pytest.mark.skip
def test_allclose(Qs):
    for q in Qs[Qs_nonnan]:
        assert quaternion.allclose(q, q, rtol=0.0, atol=0.0)
    assert quaternion.allclose(Qs[Qs_nonnan], Qs[Qs_nonnan], rtol=0.0, atol=0.0)

    for q in Qs[Qs_finitenonzero]:
        assert quaternion.allclose(q, q*(1+1e-13), rtol=1.1e-13, atol=0.0)
        assert ~quaternion.allclose(q, q*(1+1e-13), rtol=0.9e-13, atol=0.0)
        for e in [quaternion.one, quaternion.x, quaternion.y, quaternion.z]:
            assert quaternion.allclose(q, q+(1e-13*e), rtol=0.0, atol=1.1e-13)
            assert ~quaternion.allclose(q, q+(1e-13*e), rtol=0.0, atol=0.9e-13)
    assert quaternion.allclose(Qs[Qs_finitenonzero], Qs[Qs_finitenonzero]*(1+1e-13), rtol=1.1e-13, atol=0.0)
    assert ~quaternion.allclose(Qs[Qs_finitenonzero], Qs[Qs_finitenonzero]*(1+1e-13), rtol=0.9e-13, atol=0.0)
    for e in [quaternion.one, quaternion.x, quaternion.y, quaternion.z]:
        assert quaternion.allclose(Qs[Qs_finite], Qs[Qs_finite]+(1e-13*e), rtol=0.0, atol=1.1e-13)
        assert ~quaternion.allclose(Qs[Qs_finite], Qs[Qs_finite]+(1e-13*e), rtol=0.0, atol=0.9e-13)
    assert quaternion.allclose(Qs[Qs_zero], Qs[Qs_zero]*2, rtol=0.0, atol=1.1e-13)

    for qnan in Qs[Qs_nan]:
        assert ~quaternion.allclose(qnan, qnan, rtol=1.0, atol=1.0)
        for q in Qs:
            assert ~quaternion.allclose(q, qnan, rtol=1.0, atol=1.0)


@pytest.mark.skip
def test_quaternion_sqrt(Qs):
    sqrt_precision = 2.e-15
    for q in Qs[Qs_finitenonzero]:
        assert allclose(q.sqrt() * q.sqrt(), q, rtol=sqrt_precision)
        # Ensure that non-unit quaternions are handled correctly
        for s in [1, -1, 2, -2, 3.4, -3.4]:
            for r in [1, quaternion.x, quaternion.y, quaternion.z]:
                srq = s*r*q
                assert allclose(srq.sqrt() * srq.sqrt(), srq, rtol=sqrt_precision)
    # Ensure that inputs close to zero are handled gracefully
    sqrt_dbl_min = math.sqrt(np.finfo(float).tiny)
    assert dual(0, 0, 0, 2e-8*sqrt_dbl_min).sqrt() == dual(0, 0, 0, 0)
    assert dual(0, 0, 0, 0.9999*sqrt_dbl_min).sqrt() == dual(0, 0, 0, 0)
    assert dual(0, 0, 0, 1e-16*sqrt_dbl_min).sqrt() == dual(0, 0, 0, 0)
    assert dual(0, 0, 0, 1.1*sqrt_dbl_min).sqrt() != dual(0, 0, 0, 0)
    
    

@pytest.mark.skip
def test_quaternion_square(Qs):
    square_precision = 1.e-15
    for q in Qs[Qs_finite]:
        assert np.norm(q*q - q**2) < square_precision
        a = np.array([q])
        assert np.norm(a**2 - np.array([q**2])) < square_precision


@pytest.mark.skip
def test_quaternion_log_exp(Qs):
    qlogexp_precision = 4.e-15
    assert (Qs[Q].log() - Qs[Qlog]).abs() < qlogexp_precision
    assert (Qs[Q].exp() - Qs[Qexp]).abs() < qlogexp_precision
    assert (Qs[Q].log().exp() - Qs[Q]).abs() < qlogexp_precision
    assert (Qs[Q].exp().log() - Qs[Q]).abs() > qlogexp_precision  # Note order of operations!
    assert quaternion.one.log() == quaternion.zero
    assert quaternion.x.log() == (np.pi / 2) * quaternion.x
    assert quaternion.y.log() == (np.pi / 2) * quaternion.y
    assert quaternion.z.log() == (np.pi / 2) * quaternion.z
    assert (-quaternion.one).log() == (np.pi) * quaternion.x
    strict_assert(False)  # logs of interesting scalars * basis vectors
    strict_assert(False)  # logs of negative scalars


@pytest.mark.skip
def test_angle(Rs):
    angle_precision = 4.e-15
    unit_vecs = [quaternion.x, quaternion.y, quaternion.z,
                 -quaternion.x, -quaternion.y, -quaternion.z]
    for u in unit_vecs:
        for theta in np.linspace(-2 * np.pi, 2 * np.pi, num=50):
            assert abs((theta * u / 2).exp().angle() - abs(theta)) < angle_precision




# Quaternion-quaternion binary quaternion returners
@pytest.mark.xfail
def test_quaternion_copysign(Qs):
    assert False


# Quaternion-quaternion, scalar-quaternion, or quaternion-scalar binary quaternion returners



## XXX: test power better: consistency w/ exp(log), zero-norm numbers, quat**quat loops
def test_power_scalar():
    # light testing only
    qs = [dual(1.1, 0.0),
          dual(1.1, 2.2)]

    for s in [-2., -1, 0, 0.0, 1, 1.0, 2, 2.0, 5.6]:
        for q in qs:
            q_s = dual(q.real**s, s*q.imag*q.real**(s-1))
            assert allclose(q**s, q_s)


def test_quaternion_power(Qs):
    import math
    qpower_precision = 4*eps

    # FIXME: corner cases: 0**0, 1**0, 0**1 etc (b, e = 0, 0)

    # Test equivalence between scalar and real-quaternion exponentiation
    for b in [1, 1.0, 2, 2.0, 5.6]:
        for e in [1, 1.0, 2, 2.0, 4.5]:
            be = np.dual(b**e, 0)
            assert allclose(be, np.dual(b, 0)**np.dual(e, 0), rtol=qpower_precision)
            assert allclose(be, b**np.dual(e, 0), rtol=qpower_precision)
            assert allclose(be, np.dual(b, 0)**e, rtol=qpower_precision)

    # Check that exp(q) is the same as e**q
    for q in Qs[Qs_finitenonzero]:
        assert allclose(q.exp(), math.e**q, rtol=qpower_precision)
        for s in [1.0, 1, 1.2, 2.3, 3]:
            for t in [ 1.0, 1, 1.2, 2.3, 3]:
                assert allclose((s*t)**q, (s**q)*(t**q), rtol=3*qpower_precision)
                
        for s in [1.0, 1, 1.2, 2.3, 3]:
            assert allclose(s**q, (q*math.log(s)).exp(), rtol=qpower_precision)



@pytest.mark.skip
def test_quaternion_power_2(Qs):
    import math
    qpower_precision = 4*eps

    # FIXME: corner cases: 0**0, 1**0, 0**1 etc (b, e = 0, 0)

    # Test equivalence between scalar and real-quaternion exponentiation
    for b in [1, 1.0, 2, 2.0, 5.6]:
        for e in [1, 1.0, 2, 2.0, 4.5]:
            be = np.dual(b**e, 0)
            assert allclose(be, np.dual(b, 0)**np.dual(e, 0), rtol=qpower_precision)
            assert allclose(be, b**np.dual(e, 0), rtol=qpower_precision)
            assert allclose(be, np.dual(b, 0)**e, rtol=qpower_precision)

    for q in [-3*dual_one, -2*dual_one, -dual_one,
                 dual_zero, dual_one, 3*dual_one]:
        for s in [-3, -2.3, -1.2, -1.0, 1.0, 1, 1.2, 2.3, 3]:
            for t in [-3, -2.3, -1.2, -1.0, 1.0, 1, 1.2, 2.3, 3]:
                assert allclose((s*t)**q, (s**q)*(t**q), rtol=2*qpower_precision)

    # Test basic integer-exponent and additive-exponent properties
    for q in Qs[Qs_finitenonzero]:
        assert allclose(q ** 0, np.dual(1, 0), rtol=qpower_precision)
        assert allclose(q ** 0.0, np.dual(1, 0), rtol=qpower_precision)
        assert allclose(q ** np.dual(0, 0), np.dual(1, 0), rtol=qpower_precision)
        assert allclose(((q ** 0.5) * (q ** 0.5)), q, rtol=qpower_precision)
        assert allclose(q ** 1.0, q, rtol=qpower_precision)
        assert allclose(q ** 1, q, rtol=qpower_precision)
        assert allclose(q ** np.dual(1, 0), q, rtol=qpower_precision)
        assert allclose(q ** 2.0, q * q, rtol=qpower_precision)
        assert allclose(q ** 2, q * q, rtol=qpower_precision)
        assert allclose(q ** np.dual(2, 0), q * q, rtol=qpower_precision)
        assert allclose(q ** 3, q * q * q, rtol=qpower_precision)
        assert allclose(q ** -1, q.inverse(), rtol=qpower_precision)
        assert allclose(q ** -1.0, q.inverse(), rtol=qpower_precision)
        
        for s in [-3, -2.3, -1.2, -1.0, 1.0, 1, 1.2, 2.3, 3]:
            for t in [-3, -2.3, -1.2, -1.0, 1.0, 1, 1.2, 2.3, 3]:
                assert allclose(q**(s+t), (q**s)*(q**t), rtol=2*qpower_precision)
                assert allclose(q**(s-t), (q**s)/(q**t), rtol=2*qpower_precision)

    qinverse_precision = 2*eps
    for q in Qs[Qs_finitenonzero]:
        assert allclose((q ** -1.0) * q, Qs[q_1], rtol=qinverse_precision)
    for q in Qs[Qs_finitenonzero]:
        assert allclose((q ** -1) * q, Qs[q_1], rtol=qinverse_precision)
    for q in Qs[Qs_finitenonzero]:
        assert allclose((q ** Qs[q_1]), q, rtol=qpower_precision)
    strict_assert(False)  # Try more edge cases

    for q in [quaternion.x, quaternion.y, quaternion.z]:
        assert allclose(dual(math.exp(-math.pi/2), 0, 0, 0),
                        q**q, rtol=qpower_precision)
    assert allclose(dual(math.cos(math.pi/2), 0, 0, math.sin(math.pi/2)),
                    quaternion.x**quaternion.y, rtol=qpower_precision)
    assert allclose(dual(math.cos(math.pi/2), 0, -math.sin(math.pi/2), 0),
                    quaternion.x**quaternion.z, rtol=qpower_precision)
    assert allclose(dual(math.cos(math.pi/2), 0, 0, -math.sin(math.pi/2)),
                    quaternion.y**quaternion.x, rtol=qpower_precision)
    assert allclose(dual(math.cos(math.pi/2), math.sin(math.pi/2), 0, 0),
                    quaternion.y**quaternion.z, rtol=qpower_precision)
    assert allclose(dual(math.cos(math.pi/2), 0, math.sin(math.pi/2), 0),
                    quaternion.z**quaternion.x, rtol=qpower_precision)
    assert allclose(dual(math.cos(math.pi/2), -math.sin(math.pi/2), 0, 0),
                    quaternion.z**quaternion.y, rtol=qpower_precision)





@pytest.mark.xfail
def test_arrfuncs():
    # nonzero
    # copyswap
    # copyswapn
    # getitem
    # setitem
    # compare
    # argmax
    # fillwithscalar
    assert False



@pytest.mark.xfail
def test_arraydescr():
    # new
    # richcompare
    # hash
    # repr
    # str
    assert False


@pytest.mark.xfail
def test_casts():
    # FLOAT, npy_float
    # DOUBLE, npy_double
    # LONGDOUBLE, npy_longdouble
    # BOOL, npy_bool
    # BYTE, npy_byte
    # UBYTE, npy_ubyte
    # SHORT, npy_short
    # USHORT, npy_ushort
    # INT, npy_int
    # UINT, npy_uint
    # LONG, npy_long
    # ULONG, npy_ulong
    # LONGLONG, npy_longlong
    # ULONGLONG, npy_ulonglong
    # CFLOAT, npy_float
    # CDOUBLE, npy_double
    # CLONGDOUBLE, npy_longdouble
    assert False


@pytest.mark.skip
def test_ufuncs(Rs, Qs):
    np.random.seed(1234)
    assert np.allclose(np.abs(Rs), np.ones(Rs.shape), atol=1.e-14, rtol=1.e-15)
    assert np.allclose(np.abs(np.log(Rs) - np.array([r.log() for r in Rs])), np.zeros(Rs.shape), atol=1.e-14,
                       rtol=1.e-15)
    assert np.allclose(np.abs(np.exp(Rs) - np.array([r.exp() for r in Rs])), np.zeros(Rs.shape), atol=1.e-14,
                       rtol=1.e-15)
    assert np.allclose(np.abs(Rs - Rs), np.zeros(Rs.shape), atol=1.e-14, rtol=1.e-15)
    assert np.allclose(np.abs(Rs + (-Rs)), np.zeros(Rs.shape), atol=1.e-14, rtol=1.e-15)
    assert np.allclose(np.abs(np.conjugate(Rs) - np.array([r.conjugate() for r in Rs])), np.zeros(Rs.shape),
                       atol=1.e-14, rtol=1.e-15)
    assert np.all(Rs == Rs)
    assert np.all(Rs <= Rs)
    for i in range(10):
        x = np.random.uniform(-10, 10)
        assert np.allclose(np.abs(Rs * x - np.array([r * x for r in Rs])), np.zeros(Rs.shape), atol=1.e-14, rtol=1.e-15)
        # assert np.allclose( np.abs( x*Rs - np.array([r*x for r in Rs]) ), np.zeros(Rs.shape), atol=1.e-14, rtol=1.e-15)
        strict_assert(False)
        assert np.allclose(np.abs(Rs / x - np.array([r / x for r in Rs])), np.zeros(Rs.shape), atol=1.e-14, rtol=1.e-15)
        assert np.allclose(np.abs(Rs ** x - np.array([r ** x for r in Rs])), np.zeros(Rs.shape), atol=1.e-14,
                           rtol=1.e-15)
    assert np.allclose(
        np.abs(Qs[Qs_finite] + Qs[Qs_finite] - np.array([q1 + q2 for q1, q2 in zip(Qs[Qs_finite], Qs[Qs_finite])])),
        np.zeros(Qs[Qs_finite].shape), atol=1.e-14, rtol=1.e-15)
    assert np.allclose(
        np.abs(Qs[Qs_finite] - Qs[Qs_finite] - np.array([q1 - q2 for q1, q2 in zip(Qs[Qs_finite], Qs[Qs_finite])])),
        np.zeros(Qs[Qs_finite].shape), atol=1.e-14, rtol=1.e-15)
    assert np.allclose(
        np.abs(Qs[Qs_finite] * Qs[Qs_finite] - np.array([q1 * q2 for q1, q2 in zip(Qs[Qs_finite], Qs[Qs_finite])])),
        np.zeros(Qs[Qs_finite].shape), atol=1.e-14, rtol=1.e-15)
    for Q in Qs[Qs_finite]:
        assert np.allclose(np.abs(Qs[Qs_finite] * Q - np.array([q1 * Q for q1 in Qs[Qs_finite]])),
                           np.zeros(Qs[Qs_finite].shape), atol=1.e-14, rtol=1.e-15)
        # assert np.allclose( np.abs( Q*Qs[Qs_finite] - np.array([Q*q1 for q1 in Qs[Qs_finite]]) ),
        # np.zeros(Qs[Qs_finite].shape), atol=1.e-14, rtol=1.e-15)
    assert np.allclose(np.abs(Qs[Qs_finitenonzero] / Qs[Qs_finitenonzero]
                              - np.array([q1 / q2 for q1, q2 in zip(Qs[Qs_finitenonzero], Qs[Qs_finitenonzero])])),
                       np.zeros(Qs[Qs_finitenonzero].shape), atol=1.e-14, rtol=1.e-15)
    assert np.allclose(np.abs(Qs[Qs_finitenonzero] ** Qs[Qs_finitenonzero]
                              - np.array([q1 ** q2 for q1, q2 in zip(Qs[Qs_finitenonzero], Qs[Qs_finitenonzero])])),
                       np.zeros(Qs[Qs_finitenonzero].shape), atol=1.e-14, rtol=1.e-15)
    assert np.allclose(np.abs(~Qs[Qs_finitenonzero]
                              - np.array([q.inverse() for q in Qs[Qs_finitenonzero]])),
                       np.zeros(Qs[Qs_finitenonzero].shape), atol=1.e-14, rtol=1.e-15)


@pytest.mark.parametrize(
    ("ufunc",),
    [
        # Complete list obtained from from https://docs.scipy.org/doc/numpy/reference/ufuncs.html on Sep 30, 2019
        (np.add,),
        (np.subtract,),
        (np.multiply,),
        (np.divide,),
        (np.true_divide,),
        (np.floor_divide,),
        (np.negative,),
        (np.positive,),
        (np.power,),
       # pytest.param(np.power, marks=pytest.mark.skip),
        (np.absolute,),
        (np.conj,),
        (np.conjugate,),
        (np.exp,),
        (np.log,),
        (np.sqrt,),
        (np.square,),
        (np.reciprocal,),
        (np.invert,),
        (np.less,),
        (np.less_equal,),
        (np.not_equal,),
        (np.equal,),
        (np.isfinite,),
        (np.isinf,),
        (np.isnan,),
        (np.copysign,),
        pytest.param(np.logaddexp, marks=pytest.mark.xfail),
        pytest.param(np.logaddexp2, marks=pytest.mark.xfail),
        pytest.param(np.remainder, marks=pytest.mark.xfail),
        pytest.param(np.mod, marks=pytest.mark.xfail),
        pytest.param(np.fmod, marks=pytest.mark.xfail),
        pytest.param(np.divmod, marks=pytest.mark.xfail),
        pytest.param(np.fabs, marks=pytest.mark.xfail),
        pytest.param(np.rint, marks=pytest.mark.xfail),
        pytest.param(np.sign, marks=pytest.mark.xfail),
        pytest.param(np.heaviside, marks=pytest.mark.xfail),
        pytest.param(np.exp2, marks=pytest.mark.xfail),
        pytest.param(np.log2, marks=pytest.mark.xfail),
        pytest.param(np.log10, marks=pytest.mark.xfail),
        pytest.param(np.expm1, marks=pytest.mark.xfail),
        pytest.param(np.log1p, marks=pytest.mark.xfail),
        (np.cbrt,),
        pytest.param(np.gcd, marks=pytest.mark.xfail),
        pytest.param(np.lcm, marks=pytest.mark.xfail),
        (np.sin,),
        (np.cos,),
        (np.tan,),
        (np.arcsin,),
        (np.arccos,),
        (np.arctan,),
        pytest.param(np.arctan2, marks=pytest.mark.xfail),
        pytest.param(np.hypot, marks=pytest.mark.xfail),
        (np.sinh,),
        (np.cosh,),
        (np.tanh,),
        pytest.param(np.arcsinh, marks=pytest.mark.xfail),
        pytest.param(np.arccosh, marks=pytest.mark.xfail),
        pytest.param(np.arctanh, marks=pytest.mark.xfail),
        pytest.param(np.deg2rad, marks=pytest.mark.xfail),
        pytest.param(np.rad2deg, marks=pytest.mark.xfail),
        pytest.param(np.bitwise_and, marks=pytest.mark.xfail),
        pytest.param(np.bitwise_or, marks=pytest.mark.xfail),
        pytest.param(np.bitwise_xor, marks=pytest.mark.xfail),
        pytest.param(np.left_shift, marks=pytest.mark.xfail),
        pytest.param(np.right_shift, marks=pytest.mark.xfail),
        pytest.param(np.greater, marks=pytest.mark.xfail),
        pytest.param(np.greater_equal, marks=pytest.mark.xfail),
        pytest.param(np.logical_and, marks=pytest.mark.xfail),
        pytest.param(np.logical_or, marks=pytest.mark.xfail),
        pytest.param(np.logical_xor, marks=pytest.mark.xfail),
        pytest.param(np.logical_not, marks=pytest.mark.xfail),
        pytest.param(np.maximum, marks=pytest.mark.xfail),
        pytest.param(np.minimum, marks=pytest.mark.xfail),
        pytest.param(np.fmax, marks=pytest.mark.xfail),
        pytest.param(np.fmin, marks=pytest.mark.xfail),
        pytest.param(np.isnat, marks=pytest.mark.xfail),
        pytest.param(np.fabs, marks=pytest.mark.xfail),
        pytest.param(np.signbit, marks=pytest.mark.xfail),
        pytest.param(np.nextafter, marks=pytest.mark.xfail),
        pytest.param(np.spacing, marks=pytest.mark.xfail),
        pytest.param(np.modf, marks=pytest.mark.xfail),
        pytest.param(np.ldexp, marks=pytest.mark.xfail),
        pytest.param(np.frexp, marks=pytest.mark.xfail),
        pytest.param(np.fmod, marks=pytest.mark.xfail),
        pytest.param(np.floor, marks=pytest.mark.xfail),
        pytest.param(np.ceil, marks=pytest.mark.xfail),
        pytest.param(np.trunc, marks=pytest.mark.xfail),
    ],
    ids=lambda uf:uf.__name__
)
def test_ufunc_existence(ufunc):
    qarray = Qs_array[Qs_finitenonzero]
    if ufunc.nin == 1:
        result = ufunc(qarray)
    elif ufunc.nin == 2:
        result = ufunc(qarray, qarray)


def test_sin():
    # lightly test the values of sin(a + ib) = sin(a) + ib cos(a)
    r = dual(2., 3.)
    s = np.sin(r)
    assert_allclose(s.real, np.sin(r.real), atol=1e-14)
    assert_allclose(s.imag, r.imag * np.cos(r.real), atol=1e-14)


@pytest.mark.skip
def test_numpy_array_conversion(Qs):
    "Check conversions between array as quaternions and array as floats"
    # First, just check 1-d array
    Q = Qs[Qs_nonnan][:12]  # Select first 3x4=12 non-nan elements in Qs
    assert Q.dtype == np.dtype(np.dual)
    q = quaternion.as_float_array(Q)  # View as array of floats
    assert q.dtype == np.dtype(np.float)
    assert q.shape == (12, 4)  # This is the expected shape
    for j in range(12):
        for k in range(4):  # Check each component individually
            assert q[j][k] == Q[j].components[k]
    assert np.array_equal(quaternion.as_quat_array(q), Q)  # Check that we can go backwards
    # Next, see how that works if I flatten the q array
    q = q.flatten()
    assert q.dtype == np.dtype(np.float)
    assert q.shape == (48,)
    for j in range(48):
        assert q[j] == Q[j // 4].components[j % 4]
    assert np.array_equal(quaternion.as_quat_array(q), Q)  # Check that we can go backwards
    # Now, reshape into 2-d array, and re-check
    P = Q.reshape(3, 4)  # Reshape into 3x4 array of quaternions
    p = quaternion.as_float_array(P)  # View as array of floats
    assert p.shape == (3, 4, 4)  # This is the expected shape
    for j in range(3):
        for k in range(4):
            for l in range(4):  # Check each component individually
                assert p[j][k][l] == Q[4 * j + k].components[l]
    assert np.array_equal(quaternion.as_quat_array(p), P)  # Check that we can go backwards
    # Check that we get an exception if the final dimension is not divisible by 4
    with pytest.raises(ValueError):
        quaternion.as_quat_array(np.random.rand(4, 1))
    with pytest.raises(ValueError):
        quaternion.as_quat_array(np.random.rand(4, 2))
    with pytest.raises(ValueError):
        quaternion.as_quat_array(np.random.rand(4, 3))
    with pytest.raises(ValueError):
        quaternion.as_quat_array(np.random.rand(4, 5))
    with pytest.raises(ValueError):
        quaternion.as_quat_array(np.random.rand(4, 5, 3, 2, 1))
    # Finally, check that it works on non-contiguous arrays, by adding random padding and then slicing
    q = quaternion.as_float_array(Q)
    q = np.concatenate((np.random.rand(q.shape[0], 3), q, np.random.rand(q.shape[0], 3)), axis=1)
    assert np.array_equal(quaternion.as_quat_array(q[:, 3:7]), Q)




def test_numpy_save_and_load():
    import tempfile
    a = dual_as_quat_array(np.random.rand(5,3,4))
    with tempfile.TemporaryFile() as temp:
        np.save(temp, a)
        temp.seek(0)  # Only needed here to simulate closing & reopening file, per np.save docs
        b = np.load(temp).view(dtype=np.dual)
    assert np.array_equal(a, b)

@pytest.mark.skip
def test_pickle():
    import pickle
    a = dual(1.23, 4.56)
    assert pickle.loads(pickle.dumps(a)) == a


if __name__ == '__main__':
    print("The tests should be run automatically via pytest (`pip install pytest` and then just `pytest`)")




