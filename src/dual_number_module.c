// Copyright (c) 2017, Michael Boyle
// See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include "structmember.h"

#include "dual_number.h"

// The following definitions, along with `#define NPY_PY3K 1`, can
// also be found in the header <numpy/npy_3kcompat.h>.
#if PY_MAJOR_VERSION >= 3
#define PyUString_FromString PyUnicode_FromString
static NPY_INLINE int PyInt_Check(PyObject *op) {
    int overflow = 0;
    if (!PyLong_Check(op)) {
        return 0;
    }
    PyLong_AsLongAndOverflow(op, &overflow);
    return (overflow == 0);
}
#define PyInt_AsLong PyLong_AsLong
#else
#define PyUString_FromString PyString_FromString
#endif


// The basic python object holding a dual
typedef struct {
  PyObject_HEAD
  dual obval;
} PyDual;

static PyTypeObject PyDual_Type;

// This is the crucial feature that will make a dual into a
// built-in numpy data type.  We will describe its features below.
PyArray_Descr* dual_descr;


static NPY_INLINE int
PyDual_Check(PyObject* object) {
  return PyObject_IsInstance(object,(PyObject*)&PyDual_Type);
}

static PyObject*
PyDual_FromDual(dual q) {
  PyDual* p = (PyDual*)PyDual_Type.tp_alloc(&PyDual_Type,0);
  if (p) { p->obval = q; }
  return (PyObject*)p;
}

#define PyDual_AsDual(q, o)                                             \
  /* fprintf (stderr, "file %s, line %d., PyDual_AsDual\n", __FILE__, __LINE__); */ \
  if(PyDual_Check(o)) {                                                 \
    q = ((PyDual*)o)->obval;                                            \
  } else {                                                              \
    PyErr_SetString(PyExc_TypeError,                                    \
                    "Input object is not a dual.");                     \
    return NULL;                                                        \
  }

#define PyDual_AsDualPointer(q, o)                                      \
  /* fprintf (stderr, "file %s, line %d, PyDual_AsDualPointer.\n", __FILE__, __LINE__); */ \
  if(PyDual_Check(o)) {                                                 \
    q = &((PyDual*)o)->obval;                                           \
  } else {                                                              \
    PyErr_SetString(PyExc_TypeError,                                    \
                    "Input object is not a dual.");                     \
    return NULL;                                                        \
  }

static PyObject *
pydual_new(PyTypeObject *type, PyObject *NPY_UNUSED(args), PyObject *NPY_UNUSED(kwds))
{
  PyDual* self;
  self = (PyDual *)type->tp_alloc(type, 0);
  return (PyObject *)self;
}

static int
pydual_init(PyObject *self, PyObject *args, PyObject *kwds)
{
  // "A good rule of thumb is that for immutable types, all
  // initialization should take place in `tp_new`, while for mutable
  // types, most initialization should be deferred to `tp_init`."
  // ---Python 2.7.8 docs

  Py_ssize_t size = PyTuple_Size(args);
  dual* q;
  PyObject* Q = {0};
  q = &(((PyDual*)self)->obval);

  if (kwds && PyDict_Size(kwds)) {
    PyErr_SetString(PyExc_TypeError,
                    "dual constructor takes no keyword arguments");
    return -1;
  }

  q->re = 0.0;
  q->im = 0.0;

  if(size == 0) {
    return 0;
  } else if(size == 1) {
    if(PyArg_ParseTuple(args, "O", &Q) && PyDual_Check(Q)) {
      q->re = ((PyDual*)Q)->obval.re;
      q->im = ((PyDual*)Q)->obval.im;
      return 0;
    } else if(PyArg_ParseTuple(args, "d", &q->re)) {
      return 0;
    }
  } else if(size == 2 && PyArg_ParseTuple(args, "dd", &q->re, &q->im)) {
    return 0;
  }

  PyErr_SetString(PyExc_TypeError,
                  "dual constructor takes zero, one, or two float arguments, or a single dual");
  return -1;
}

#define UNARY_BOOL_RETURNER(name)                                       \
  static PyObject*                                                      \
  pydual_##name(PyObject* a, PyObject* NPY_UNUSED(b)) {                 \
    dual q = {0.0, 0.0}                ;                                \
    PyDual_AsDual(q, a);                                                \
    return PyBool_FromLong(dual_##name(q));                             \
  }
UNARY_BOOL_RETURNER(nonzero)
UNARY_BOOL_RETURNER(isnan)
UNARY_BOOL_RETURNER(isinf)
UNARY_BOOL_RETURNER(isfinite)

#define BINARY_BOOL_RETURNER(name)                                      \
  static PyObject*                                                      \
  pydual_##name(PyObject* a, PyObject* b) {                             \
    dual p = {0.0, 0.0};                                                \
    dual q = {0.0, 0.0};                                                \
    PyDual_AsDual(p, a);                                                \
    PyDual_AsDual(q, b);                                                \
    return PyBool_FromLong(dual_##name(p,q));                           \
  }
BINARY_BOOL_RETURNER(equal)
BINARY_BOOL_RETURNER(not_equal)
BINARY_BOOL_RETURNER(less)
BINARY_BOOL_RETURNER(greater)
BINARY_BOOL_RETURNER(less_equal)
BINARY_BOOL_RETURNER(greater_equal)


#define UNARY_FLOAT_RETURNER(name)                                      \
  static PyObject*                                                      \
  pydual_##name(PyObject* a, PyObject* NPY_UNUSED(b)) {                 \
    dual q = {0.0, 0.0};                                                \
    PyDual_AsDual(q, a);                                                \
    return PyFloat_FromDouble(dual_##name(q));                          \
  }
UNARY_FLOAT_RETURNER(absolute)
UNARY_FLOAT_RETURNER(norm)
UNARY_FLOAT_RETURNER(angle)

#define UNARY_QUATERNION_RETURNER(name)                                 \
  static PyObject*                                                      \
  pydual_##name(PyObject* a, PyObject* NPY_UNUSED(b)) {                 \
    dual q = {0.0, 0.0};                                                \
    PyDual_AsDual(q, a);                                                \
    return PyDual_FromDual(dual_##name(q));                             \
  }
UNARY_QUATERNION_RETURNER(negative)
UNARY_QUATERNION_RETURNER(conjugate)
UNARY_QUATERNION_RETURNER(inverse)
UNARY_QUATERNION_RETURNER(sqrt)
UNARY_QUATERNION_RETURNER(square)
UNARY_QUATERNION_RETURNER(log)
UNARY_QUATERNION_RETURNER(exp)
UNARY_QUATERNION_RETURNER(normalized)

static PyObject*
pydual_positive(PyObject* self, PyObject* NPY_UNUSED(b)) {
  Py_INCREF(self);
  return self;
}

#define QQ_BINARY_QUATERNION_RETURNER(name)                             \
  static PyObject*                                                      \
  pydual_##name(PyObject* a, PyObject* b) {                             \
    dual p = {0.0, 0.0};                                                \
    dual q = {0.0, 0.0};                                                \
    PyDual_AsDual(p, a);                                                \
    PyDual_AsDual(q, b);                                                \
    return PyDual_FromDual(dual_##name(p,q));                           \
  }
/* QQ_BINARY_QUATERNION_RETURNER(add) */
/* QQ_BINARY_QUATERNION_RETURNER(subtract) */
QQ_BINARY_QUATERNION_RETURNER(copysign)

#define QQ_QS_SQ_BINARY_QUATERNION_RETURNER_FULL(fake_name, name)       \
static PyObject*                                                        \
pydual_##fake_name##_array_operator(PyObject* a, PyObject* b) {         \
  NpyIter *iter;                                                        \
  NpyIter_IterNextFunc *iternext;                                       \
  PyArrayObject *op[2];                                                 \
  PyObject *ret;                                                        \
  npy_uint32 flags;                                                     \
  npy_uint32 op_flags[2];                                               \
  PyArray_Descr *op_dtypes[2];                                          \
  npy_intp itemsize, *innersizeptr, innerstride;                        \
  char **dataptrarray;                                                  \
  char *src, *dst;                                                      \
  dual p = {0.0, 0.0 };                                                  \
  PyDual_AsDual(p, a);                                                  \
  flags = NPY_ITER_EXTERNAL_LOOP;                                       \
  op[0] = (PyArrayObject *) b;                                          \
  op[1] = NULL;                                                         \
  op_flags[0] = NPY_ITER_READONLY;                                      \
  op_flags[1] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;                 \
  op_dtypes[0] = PyArray_DESCR((PyArrayObject*) b);                     \
  op_dtypes[1] = dual_descr;                                            \
  iter = NpyIter_MultiNew(2, op, flags, NPY_KEEPORDER, NPY_NO_CASTING, op_flags, op_dtypes); \
  if (iter == NULL) {                                                   \
    return NULL;                                                        \
  }                                                                     \
  iternext = NpyIter_GetIterNext(iter, NULL);                           \
  innerstride = NpyIter_GetInnerStrideArray(iter)[0];                   \
  itemsize = NpyIter_GetDescrArray(iter)[1]->elsize;                    \
  innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);                     \
  dataptrarray = NpyIter_GetDataPtrArray(iter);                         \
  if(PyArray_EquivTypes(PyArray_DESCR((PyArrayObject*) b), dual_descr)) { \
    npy_intp i;                                                         \
    do {                                                                \
      npy_intp size = *innersizeptr;                                    \
      src = dataptrarray[0];                                            \
      dst = dataptrarray[1];                                            \
      for(i = 0; i < size; i++, src += innerstride, dst += itemsize) {  \
        *((dual *) dst) = dual_##name(p, *((dual *) src));              \
      }                                                                 \
    } while (iternext(iter));                                           \
  } else if(PyArray_ISFLOAT((PyArrayObject*) b)) {                      \
    npy_intp i;                                                         \
    do {                                                                \
      npy_intp size = *innersizeptr;                                    \
      src = dataptrarray[0];                                            \
      dst = dataptrarray[1];                                            \
      for(i = 0; i < size; i++, src += innerstride, dst += itemsize) {  \
        *(dual *) dst = dual_##name##_scalar(p, *((double *) src));     \
      }                                                                 \
    } while (iternext(iter));                                           \
  } else if(PyArray_ISINTEGER((PyArrayObject*) b)) {                    \
    npy_intp i;                                                         \
    do {                                                                \
      npy_intp size = *innersizeptr;                                    \
      src = dataptrarray[0];                                            \
      dst = dataptrarray[1];                                            \
      for(i = 0; i < size; i++, src += innerstride, dst += itemsize) {  \
        *((dual *) dst) = dual_##name##_scalar(p, *((int *) src));      \
      }                                                                 \
    } while (iternext(iter));                                           \
  } else {                                                              \
    NpyIter_Deallocate(iter);                                           \
    return NULL;                                                        \
  }                                                                     \
  ret = (PyObject *) NpyIter_GetOperandArray(iter)[1];                  \
  Py_INCREF(ret);                                                       \
  if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {                        \
    Py_DECREF(ret);                                                     \
    return NULL;                                                        \
  }                                                                     \
  return ret;                                                           \
}                                                                       \
static PyObject*                                                        \
pydual_##fake_name(PyObject* a, PyObject* b) {                          \
  /* PyObject *a_type, *a_repr, *b_type, *b_repr, *a_repr2, *b_repr2;    \ */ \
  /* char* a_char, b_char, a_char2, b_char2;                             \ */ \
  npy_int64 val64;                                                     \
  npy_int32 val32;                                                     \
  dual p = {0.0, 0.0};                                                 \
  if(PyArray_Check(b)) { return pydual_##fake_name##_array_operator(a, b); } \
  if(PyFloat_Check(a) && PyDual_Check(b)) {                            \
    return PyDual_FromDual(dual_scalar_##name(PyFloat_AsDouble(a), ((PyDual*)b)->obval)); \
  }                                                                    \
  if(PyInt_Check(a) && PyDual_Check(b)) {                              \
    return PyDual_FromDual(dual_scalar_##name(PyInt_AsLong(a), ((PyDual*)b)->obval)); \
  }                                                                    \
  PyDual_AsDual(p, a);                                                 \
  if(PyDual_Check(b)) {                                                \
    return PyDual_FromDual(dual_##name(p,((PyDual*)b)->obval));        \
  } else if(PyFloat_Check(b)) {                                        \
    return PyDual_FromDual(dual_##name##_scalar(p,PyFloat_AsDouble(b))); \
  } else if(PyInt_Check(b)) {                                          \
    return PyDual_FromDual(dual_##name##_scalar(p,PyInt_AsLong(b)));   \
  } else if(PyObject_TypeCheck(b, &PyInt64ArrType_Type)) {             \
    PyArray_ScalarAsCtype(b, &val64);                                  \
    return PyDual_FromDual(dual_##name##_scalar(p, val64)); \
  } else if(PyObject_TypeCheck(b, &PyInt32ArrType_Type)) {             \
    PyArray_ScalarAsCtype(b, &val32);                                  \
    return PyDual_FromDual(dual_##name##_scalar(p, val32)); \
  }                                                                    \
  PyErr_SetString(PyExc_TypeError, "Binary operation involving dual and \\neither float nor dual."); \
  return NULL;                                                          \
}
#define QQ_QS_SQ_BINARY_QUATERNION_RETURNER(name) QQ_QS_SQ_BINARY_QUATERNION_RETURNER_FULL(name, name)
QQ_QS_SQ_BINARY_QUATERNION_RETURNER(add)
QQ_QS_SQ_BINARY_QUATERNION_RETURNER(subtract)
QQ_QS_SQ_BINARY_QUATERNION_RETURNER(multiply)
QQ_QS_SQ_BINARY_QUATERNION_RETURNER(divide)
/* QQ_QS_SQ_BINARY_QUATERNION_RETURNER_FULL(true_divide, divide) */
/* QQ_QS_SQ_BINARY_QUATERNION_RETURNER_FULL(floor_divide, divide) */
QQ_QS_SQ_BINARY_QUATERNION_RETURNER(power)

#define QQ_QS_SQ_BINARY_QUATERNION_INPLACE_FULL(fake_name, name)        \
  static PyObject*                                                      \
  pydual_inplace_##fake_name(PyObject* a, PyObject* b) {          \
    dual* p = {0};                                                \
    /* fprintf (stderr, "file %s, line %d, pydual_inplace_"#fake_name"(PyObject* a, PyObject* b).\n", __FILE__, __LINE__); \ */ \
    if(PyFloat_Check(a) || PyInt_Check(a)) {                            \
      PyErr_SetString(PyExc_TypeError, "Cannot in-place "#fake_name" a scalar by a dual; should be handled by python."); \
      return NULL;                                                      \
    }                                                                   \
    PyDual_AsDualPointer(p, a);                             \
    if(PyDual_Check(b)) {                                         \
      dual_inplace_##name(p,((PyDual*)b)->obval);           \
      Py_INCREF(a);                                                     \
      return a;                                                         \
    } else if(PyFloat_Check(b)) {                                       \
      dual_inplace_##name##_scalar(p,PyFloat_AsDouble(b));        \
      Py_INCREF(a);                                                     \
      return a;                                                         \
    } else if(PyInt_Check(b)) {                                         \
      dual_inplace_##name##_scalar(p,PyInt_AsLong(b));            \
      Py_INCREF(a);                                                     \
      return a;                                                         \
    }                                                                   \
    PyErr_SetString(PyExc_TypeError, "Binary in-place operation involving dual and neither float nor dual."); \
    return NULL;                                                        \
  }
#define QQ_QS_SQ_BINARY_QUATERNION_INPLACE(name) QQ_QS_SQ_BINARY_QUATERNION_INPLACE_FULL(name, name)
QQ_QS_SQ_BINARY_QUATERNION_INPLACE(add)
QQ_QS_SQ_BINARY_QUATERNION_INPLACE(subtract)
QQ_QS_SQ_BINARY_QUATERNION_INPLACE(multiply)
QQ_QS_SQ_BINARY_QUATERNION_INPLACE(divide)
/* QQ_QS_SQ_BINARY_QUATERNION_INPLACE_FULL(true_divide, divide) */
/* QQ_QS_SQ_BINARY_QUATERNION_INPLACE_FULL(floor_divide, divide) */
QQ_QS_SQ_BINARY_QUATERNION_INPLACE(power)

static PyObject *
pydual__reduce(PyDual* self)
{
  /* printf("\n\n\nI'm trying, most of all!\n\n\n"); */
  return Py_BuildValue("O(OO)", Py_TYPE(self),
                       PyFloat_FromDouble(self->obval.re), PyFloat_FromDouble(self->obval.im));
}

static PyObject *
pydual_getstate(PyDual* self, PyObject* args)
{
  /* printf("\n\n\nI'm Trying, OKAY?\n\n\n"); */
  if (!PyArg_ParseTuple(args, ":getstate"))
    return NULL;
  return Py_BuildValue("OO",
                       PyFloat_FromDouble(self->obval.re), PyFloat_FromDouble(self->obval.im));
}

static PyObject *
pydual_setstate(PyDual* self, PyObject* args)
{
  /* printf("\n\n\nI'm Trying, TOO!\n\n\n"); */
  dual* q;
  q = &(self->obval);

  if (!PyArg_ParseTuple(args, "dd:setstate", &q->re, &q->im)) {
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}


// This is an array of methods (member functions) that will be
// available to use on the dual objects in python.  This is
// packaged up here, and will be used in the `tp_methods` field when
// definining the PyDual_Type below.
PyMethodDef pydual_methods[] = {
  // Unary bool returners
  {"nonzero", pydual_nonzero, METH_NOARGS,
   "True if the dual has all zero components"},
  {"isnan", pydual_isnan, METH_NOARGS,
   "True if the dual has any NAN components"},
  {"isinf", pydual_isinf, METH_NOARGS,
   "True if the dual has any INF components"},
  {"isfinite", pydual_isfinite, METH_NOARGS,
   "True if the dual has all finite components"},

  // Binary bool returners
  {"equal", pydual_equal, METH_O,
   "True if the duals are PRECISELY equal"},
  {"not_equal", pydual_not_equal, METH_O,
   "True if the duals are not PRECISELY equal"},
  {"less", pydual_less, METH_O,
   "Strict dictionary ordering"},
  {"greater", pydual_greater, METH_O,
   "Strict dictionary ordering"},
  {"less_equal", pydual_less_equal, METH_O,
   "Dictionary ordering"},
  {"greater_equal", pydual_greater_equal, METH_O,
   "Dictionary ordering"},


  // Unary float returners
  {"absolute", pydual_absolute, METH_NOARGS,
   "Absolute value of dual"},
  {"abs", pydual_absolute, METH_NOARGS,
   "Absolute value (Euclidean norm) of dual"},
  {"norm", pydual_norm, METH_NOARGS,
   "Cayley norm (square of the absolute value) of dual"},
  //{"angle", pydual_angle, METH_NOARGS,
  // "Angle through which rotor rotates"},

  // Unary dual returners
  // {"negative", pydual_negative, METH_NOARGS,
  //  "Return the negated dual"},
  // {"positive", pydual_positive, METH_NOARGS,
  //  "Return the dual itself"},
  {"conjugate", pydual_conjugate, METH_NOARGS,
   "Return the complex conjugate of the dual"},
  {"conj", pydual_conjugate, METH_NOARGS,
   "Return the complex conjugate of the dual"},
  {"inverse", pydual_inverse, METH_NOARGS,
   "Return the inverse of the dual"},
  {"reciprocal", pydual_inverse, METH_NOARGS,
   "Return the reciprocal of the dual"},
  {"sqrt", pydual_sqrt, METH_NOARGS,
   "Return the square-root of the dual"},
  {"square", pydual_square, METH_NOARGS,
   "Return the square of the dual"},
  {"log", pydual_log, METH_NOARGS,
   "Return the logarithm (base e) of the dual"},
  {"exp", pydual_exp, METH_NOARGS,
   "Return the exponential of the dual (e**q)"},
  {"normalized", pydual_normalized, METH_NOARGS,
   "Return a normalized copy of the dual"},

  // Dual-dual binary dual returners
  // {"add", pydual_add, METH_O,
  //  "Componentwise addition"},
  // {"subtract", pydual_subtract, METH_O,
  //  "Componentwise subtraction"},
  {"copysign", pydual_copysign, METH_O,
   "Componentwise copysign"},

  // Dual-dual or dual-scalar binary dual returners
  // {"multiply", pydual_multiply, METH_O,
  //  "Standard (geometric) dual product"},
  // {"divide", pydual_divide, METH_O,
  //  "Standard (geometric) dual division"},
  // {"power", pydual_power, METH_O,
  //  "q.power(p) = (q.log() * p).exp()"},

  {"__reduce__", (PyCFunction)pydual__reduce, METH_NOARGS,
   "Return state information for pickling."},
  {"__getstate__", (PyCFunction)pydual_getstate, METH_VARARGS,
   "Return state information for pickling."},
  {"__setstate__", (PyCFunction)pydual_setstate, METH_VARARGS,
   "Reconstruct state information from pickle."},

  {NULL, NULL, 0, NULL}
};

static PyObject* pydual_num_power(PyObject* a, PyObject* b, PyObject *c) { (void) c; return pydual_power(a,b); }
static PyObject* pydual_num_inplace_power(PyObject* a, PyObject* b, PyObject *c) { (void) c; return pydual_inplace_power(a,b); }
static PyObject* pydual_num_negative(PyObject* a) { return pydual_negative(a,NULL); }
static PyObject* pydual_num_positive(PyObject* a) { return pydual_positive(a,NULL); }
static PyObject* pydual_num_absolute(PyObject* a) { return pydual_absolute(a,NULL); }
static PyObject* pydual_num_inverse(PyObject* a) { return pydual_inverse(a,NULL); }
static int pydual_num_nonzero(PyObject* a) {
  dual q = ((PyDual*)a)->obval;
  return dual_nonzero(q);
}
#define CANNOT_CONVERT(target)                                          \
  static PyObject* pydual_convert_##target(PyObject* a) {         \
    PyErr_SetString(PyExc_TypeError, "Cannot convert dual to " #target); \
    return NULL;                                                        \
  }
CANNOT_CONVERT(int)
CANNOT_CONVERT(float)
#if PY_MAJOR_VERSION < 3
CANNOT_CONVERT(long)
CANNOT_CONVERT(oct)
CANNOT_CONVERT(hex)
#endif

static PyNumberMethods pydual_as_number = {
  pydual_add,               // nb_add
  pydual_subtract,          // nb_subtract
  pydual_multiply,          // nb_multiply
  #if PY_MAJOR_VERSION < 3
  pydual_divide,            // nb_divide
  #endif
  0,                              // nb_remainder
  0,                              // nb_divmod
  pydual_num_power,         // nb_power
  pydual_num_negative,      // nb_negative
  pydual_num_positive,      // nb_positive
  pydual_num_absolute,      // nb_absolute
  pydual_num_nonzero,       // nb_nonzero
  pydual_num_inverse,       // nb_invert
  0,                              // nb_lshift
  0,                              // nb_rshift
  0,                              // nb_and
  0,                              // nb_xor
  0,                              // nb_or
  #if PY_MAJOR_VERSION < 3
  0,                              // nb_coerce
  #endif
  pydual_convert_int,       // nb_int
  #if PY_MAJOR_VERSION >= 3
  0,                              // nb_reserved
  #else
  pydual_convert_long,      // nb_long
  #endif
  pydual_convert_float,     // nb_float
  #if PY_MAJOR_VERSION < 3
  pydual_convert_oct,       // nb_oct
  pydual_convert_hex,       // nb_hex
  #endif
  pydual_inplace_add,       // nb_inplace_add
  pydual_inplace_subtract,  // nb_inplace_subtract
  pydual_inplace_multiply,  // nb_inplace_multiply
  #if PY_MAJOR_VERSION < 3
  pydual_inplace_divide,    // nb_inplace_divide
  #endif
  0,                              // nb_inplace_remainder
  pydual_num_inplace_power, // nb_inplace_power
  0,                              // nb_inplace_lshift
  0,                              // nb_inplace_rshift
  0,                              // nb_inplace_and
  0,                              // nb_inplace_xor
  0,                              // nb_inplace_or
  pydual_divide,            // nb_floor_divide
  pydual_divide,            // nb_true_divide
  pydual_inplace_divide,    // nb_inplace_floor_divide
  pydual_inplace_divide,    // nb_inplace_true_divide
  0,                              // nb_index
  #if PY_MAJOR_VERSION >= 3
  #if PY_MINOR_VERSION >= 5
  0,                              // nb_matrix_multiply
  0,                              // nb_inplace_matrix_multiply
  #endif
  #endif
};


// This is an array of members (member data) that will be available to
// use on the dual objects in python.  This is packaged up here,
// and will be used in the `tp_members` field when definining the
// PyDual_Type below.
PyMemberDef pydual_members[] = {
  {"real", T_DOUBLE, offsetof(PyDual, obval.re), 0,
   "The real component of the dual"},
  {"imag", T_DOUBLE, offsetof(PyDual, obval.im), 0,
   "The imaginary component of the dual"},
  {NULL, 0, 0, 0, NULL}
};




// This will be defined as a member function on the dual
// objects, so that calling "components" will return a numpy array
// with the components of the dual.
static PyObject *
pydual_get_components(PyObject *self, void *NPY_UNUSED(closure))
{
  dual *q = &((PyDual *)self)->obval;
  int nd = 1;
  npy_intp dims[1] = { 2 };
  int typenum = NPY_DOUBLE;
  PyObject* components = PyArray_SimpleNewFromData(nd, dims, typenum, &(q->re));
  Py_INCREF(self);
  PyArray_SetBaseObject((PyArrayObject*)components, self);
  return components;
}

// This will be defined as a member function on the dual
// objects, so that calling `q.components = [1,2]`, for example,
// will set the components appropriately.
static int
pydual_set_components(PyObject *self, PyObject *value, void *NPY_UNUSED(closure)){
  PyObject *element;
  dual *q = &((PyDual *)self)->obval;
  if (value == NULL) {
    PyErr_SetString(PyExc_ValueError, "Cannot set dual to empty value");
    return -1;
  }
  if (! (PySequence_Check(value) && PySequence_Size(value)==2) ) {
    PyErr_SetString(PyExc_TypeError,
                    "A dual's components must be set to something of length 2");
    return -1;
  }
  element = PySequence_GetItem(value, 0);
  if(element == NULL) { return -1; } /* Not a sequence, or other failure */
  q->re = PyFloat_AsDouble(element);
  Py_DECREF(element);
  element = PySequence_GetItem(value, 1);
  if(element == NULL) { return -1; } /* Not a sequence, or other failure */
  q->im = PyFloat_AsDouble(element);
  Py_DECREF(element);
  return 0;
}

// This collects the methods for getting and setting elements of the
// dual.  This is packaged up here, and will be used in the
// `tp_getset` field when definining the PyDual_Type
// below.
PyGetSetDef pydual_getset[] = {
  {"components", pydual_get_components, pydual_set_components,
   "The components (w,x,y,z) of the dual as a numpy array", NULL},
  {NULL, NULL, NULL, NULL, NULL}
};



static PyObject*
pydual_richcompare(PyObject* a, PyObject* b, int op)
{
  dual x = {0.0, 0.0, 0.0, 0.0};
  dual y = {0.0, 0.0, 0.0, 0.0};
  int result = 0;
  PyDual_AsDual(x,a);
  PyDual_AsDual(y,b);
  #define COMPARISONOP(py,op) case py: result = dual_##op(x,y); break;
  switch (op) {
    COMPARISONOP(Py_LT,less)
    COMPARISONOP(Py_LE,less_equal)
    COMPARISONOP(Py_EQ,equal)
    COMPARISONOP(Py_NE,not_equal)
    COMPARISONOP(Py_GT,greater)
    COMPARISONOP(Py_GE,greater_equal)
  };
  #undef COMPARISONOP
  return PyBool_FromLong(result);
}

static long
pydual_hash(PyObject *o)
{
  dual q = ((PyDual *)o)->obval;
  long value = 0x456789;
  value = (10000004 * value) ^ _Py_HashDouble(q.re);
  value = (10000004 * value) ^ _Py_HashDouble(q.im);
  if (value == -1)
    value = -2;
  return value;
}

static PyObject *
pydual_repr(PyObject *o)
{
  char str[128];
  dual q = ((PyDual *)o)->obval;
  sprintf(str, "dual(%.15g, %.15g)", q.re, q.im);
  return PyUString_FromString(str);
}

static PyObject *
pydual_str(PyObject *o)
{
  char str[128];
  dual q = ((PyDual *)o)->obval;
  sprintf(str, "dual(%.15g, %.15g)", q.re, q.im);
  return PyUString_FromString(str);
}


// This establishes the dual as a python object (not yet a numpy
// scalar type).  The name may be a little counterintuitive; the idea
// is that this will be a type that can be used as an array dtype.
// Note that many of the slots below will be filled later, after the
// corresponding functions are defined.
static PyTypeObject PyDual_Type = {
#if PY_MAJOR_VERSION >= 3
  PyVarObject_HEAD_INIT(NULL, 0)
#else
  PyObject_HEAD_INIT(NULL)
  0,                                          // ob_size
#endif
  "dual.dual",                    // tp_name
  sizeof(PyDual),                       // tp_basicsize
  0,                                          // tp_itemsize
  0,                                          // tp_dealloc
  0,                                          // tp_print
  0,                                          // tp_getattr
  0,                                          // tp_setattr
#if PY_MAJOR_VERSION >= 3
  0,                                          // tp_reserved
#else
  0,                                          // tp_compare
#endif
  pydual_repr,                          // tp_repr
  &pydual_as_number,                    // tp_as_number
  0,                                          // tp_as_sequence
  0,                                          // tp_as_mapping
  pydual_hash,                          // tp_hash
  0,                                          // tp_call
  pydual_str,                           // tp_str
  0,                                          // tp_getattro
  0,                                          // tp_setattro
  0,                                          // tp_as_buffer
#if PY_MAJOR_VERSION >= 3
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   // tp_flags
#else
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES, // tp_flags
#endif
  "Floating-point dual numbers",        // tp_doc
  0,                                          // tp_traverse
  0,                                          // tp_clear
  pydual_richcompare,                   // tp_richcompare
  0,                                          // tp_weaklistoffset
  0,                                          // tp_iter
  0,                                          // tp_iternext
  pydual_methods,                       // tp_methods
  pydual_members,                       // tp_members
  pydual_getset,                        // tp_getset
  0,                                          // tp_base; will be reset to &PyGenericArrType_Type after numpy import
  0,                                          // tp_dict
  0,                                          // tp_descr_get
  0,                                          // tp_descr_set
  0,                                          // tp_dictoffset
  pydual_init,                          // tp_init
  0,                                          // tp_alloc
  pydual_new,                           // tp_new
  0,                                          // tp_free
  0,                                          // tp_is_gc
  0,                                          // tp_bases
  0,                                          // tp_mro
  0,                                          // tp_cache
  0,                                          // tp_subclasses
  0,                                          // tp_weaklist
  0,                                          // tp_del
#if PY_VERSION_HEX >= 0x02060000
  0,                                          // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x030400a1
  0,                                          // tp_finalize
#endif
};

// Functions implementing internal features. Not all of these function
// pointers must be defined for a given type. The required members are
// nonzero, copyswap, copyswapn, setitem, getitem, and cast.
static PyArray_ArrFuncs _PyDual_ArrFuncs;

static npy_bool
QUATERNION_nonzero (char *ip, PyArrayObject *ap)
{
  dual q;
  dual zero = {0,0};
  if (ap == NULL || PyArray_ISBEHAVED_RO(ap)) {
    q = *(dual *)ip;
  }
  else {
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_DOUBLE);
    descr->f->copyswap(&q.re, ip, !PyArray_ISNOTSWAPPED(ap), NULL);
    descr->f->copyswap(&q.im, ip+8, !PyArray_ISNOTSWAPPED(ap), NULL);
    Py_DECREF(descr);
  }
  return (npy_bool) !dual_equal(q, zero);
}

static void
QUATERNION_copyswap(dual *dst, dual *src,
                    int swap, void *NPY_UNUSED(arr))
{
  PyArray_Descr *descr;
  descr = PyArray_DescrFromType(NPY_DOUBLE);
  descr->f->copyswapn(dst, sizeof(double), src, sizeof(double), 2, swap, NULL);
  Py_DECREF(descr);
}

static void
QUATERNION_copyswapn(dual *dst, npy_intp dstride,
                     dual *src, npy_intp sstride,
                     npy_intp n, int swap, void *NPY_UNUSED(arr))
{
  PyArray_Descr *descr;
  descr = PyArray_DescrFromType(NPY_DOUBLE);
  descr->f->copyswapn(&dst->re, dstride, &src->re, sstride, n, swap, NULL);
  descr->f->copyswapn(&dst->im, dstride, &src->im, sstride, n, swap, NULL);
  Py_DECREF(descr);
}

static int QUATERNION_setitem(PyObject* item, dual* qp, void* NPY_UNUSED(ap))
{
  PyObject *element;
  if(PyDual_Check(item)) {
    memcpy(qp,&(((PyDual *)item)->obval),sizeof(dual));
  } else if(PySequence_Check(item) && PySequence_Length(item)==2) {
    element = PySequence_GetItem(item, 0);
    if(element == NULL) { return -1; } /* Not a sequence, or other failure */
    qp->re = PyFloat_AsDouble(element);
    Py_DECREF(element);
    element = PySequence_GetItem(item, 1);
    if(element == NULL) { return -1; } /* Not a sequence, or other failure */
    qp->im = PyFloat_AsDouble(element);
    Py_DECREF(element);
  } else {
    PyErr_SetString(PyExc_TypeError,
                    "Unknown input to QUATERNION_setitem");
    return -1;
  }
  return 0;
}

// When a numpy array of dtype=dual is indexed, this function is
// called, returning a new dual object with a copy of the
// data... sometimes...
static PyObject *
QUATERNION_getitem(void* data, void* NPY_UNUSED(arr))
{
  dual q;
  memcpy(&q,data,sizeof(dual));
  return PyDual_FromDual(q);
}

/*** DUAL_DBL
static int
QUATERNION_compare(dual *pa, dual *pb, PyArrayObject *NPY_UNUSED(ap))
{
  dual a = *pa, b = *pb;
  npy_bool anan, bnan;
  int ret;

  anan = dual_isnan(a);
  bnan = dual_isnan(b);

  if (anan) {
    ret = bnan ? 0 : -1;
  } else if (bnan) {
    ret = 1;
  } else if(dual_less(a, b)) {
    ret = -1;
  } else if(dual_less(b, a)) {
    ret = 1;
  } else {
    ret = 0;
  }

  return ret;
}
***/

static int
QUATERNION_argmax(dual *ip, npy_intp n, npy_intp *max_ind, PyArrayObject *NPY_UNUSED(aip))
{
  npy_intp i;
  dual mp = *ip;

  *max_ind = 0;

/*** DUAL_DBL

  if (dual_isnan(mp)) {
    // nan encountered; it's maximal
    return 0;
  }

  for (i = 1; i < n; i++) {
    ip++;
    //Propagate nans, similarly as max() and min()
    if (!(dual_less_equal(*ip, mp))) {  // negated, for correct nan handling
      mp = *ip;
      *max_ind = i;
      if (dual_isnan(mp)) {
        // nan encountered, it's maximal
        break;
      }
    }
  }
  
***/
  return 0;
}

static void
QUATERNION_fillwithscalar(dual *buffer, npy_intp length, dual *value, void *NPY_UNUSED(ignored))
{
  npy_intp i;
  dual val = *value;

  for (i = 0; i < length; ++i) {
    buffer[i] = val;
  }
}

// This is a macro (followed by applications of the macro) that cast
// the input types to standard duals with only a nonzero scalar
// part.
#define MAKE_T_TO_QUATERNION(TYPE, type)                                \
  static void                                                           \
  TYPE ## _to_dual(type *ip, dual *op, npy_intp n,          \
                         PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
  {                                                                     \
    while (n--) {                                                       \
      op->re = (double)(*ip++);                                          \
      op->im = 0;                                                        \
      op++;                                                             \
    }                                                                   \
  }
MAKE_T_TO_QUATERNION(FLOAT, npy_float);
MAKE_T_TO_QUATERNION(DOUBLE, npy_double);
MAKE_T_TO_QUATERNION(LONGDOUBLE, npy_longdouble);
MAKE_T_TO_QUATERNION(BOOL, npy_bool);
MAKE_T_TO_QUATERNION(BYTE, npy_byte);
MAKE_T_TO_QUATERNION(UBYTE, npy_ubyte);
MAKE_T_TO_QUATERNION(SHORT, npy_short);
MAKE_T_TO_QUATERNION(USHORT, npy_ushort);
MAKE_T_TO_QUATERNION(INT, npy_int);
MAKE_T_TO_QUATERNION(UINT, npy_uint);
MAKE_T_TO_QUATERNION(LONG, npy_long);
MAKE_T_TO_QUATERNION(ULONG, npy_ulong);
MAKE_T_TO_QUATERNION(LONGLONG, npy_longlong);
MAKE_T_TO_QUATERNION(ULONGLONG, npy_ulonglong);

/*** DUAL_DBL

// This is a macro (followed by applications of the macro) that cast
// the input complex types to standard duals with only the first
// two components nonzero.  This doesn't make a whole lot of sense to
// me, and may be removed in the future.
#define MAKE_CT_TO_QUATERNION(TYPE, type)                               \
  static void                                                           \
  TYPE ## _to_dual(type *ip, dual *op, npy_intp n,          \
                         PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
  {                                                                     \
    while (n--) {                                                       \
      op-> = (double)(*ip++);                                          \
      op->x = (double)(*ip++);                                          \
      op->y = 0;                                                        \
      op->z = 0;                                                        \
    }                                                                   \
  }
MAKE_CT_TO_QUATERNION(CFLOAT, npy_float);
MAKE_CT_TO_QUATERNION(CDOUBLE, npy_double);
MAKE_CT_TO_QUATERNION(CLONGDOUBLE, npy_longdouble);

***/

static void register_cast_function(int sourceType, int destType, PyArray_VectorUnaryFunc *castfunc)
{
  PyArray_Descr *descr = PyArray_DescrFromType(sourceType);
  PyArray_RegisterCastFunc(descr, destType, castfunc);
  PyArray_RegisterCanCast(descr, destType, NPY_NOSCALAR);
  Py_DECREF(descr);
}


// This is a macro that will be used to define the various basic unary
// dual functions, so that they can be applied quickly to a
// numpy array of duals.
#define UNARY_GEN_UFUNC(ufunc_name, func_name, ret_type)        \
  static void                                                           \
  dual_##ufunc_name##_ufunc(char** args, npy_intp* dimensions,    \
                                  npy_intp* steps, void* NPY_UNUSED(data)) { \
    /* fprintf (stderr, "file %s, line %d, dual_%s_ufunc.\n", __FILE__, __LINE__, #ufunc_name); */ \
    char *ip1 = args[0], *op1 = args[1];                                \
    npy_intp is1 = steps[0], os1 = steps[1];                            \
    npy_intp n = dimensions[0];                                         \
    npy_intp i;                                                         \
    for(i = 0; i < n; i++, ip1 += is1, op1 += os1){                     \
      const dual in1 = *(dual *)ip1;                        \
      *((ret_type *)op1) = dual_##func_name(in1);};}
#define UNARY_UFUNC(name, ret_type) \
  UNARY_GEN_UFUNC(name, name, ret_type)
// And these all do the work mentioned above, using the macro
UNARY_UFUNC(isnan, npy_bool)
UNARY_UFUNC(isinf, npy_bool)
UNARY_UFUNC(isfinite, npy_bool)
UNARY_UFUNC(norm, npy_double)
UNARY_UFUNC(absolute, npy_double)
UNARY_UFUNC(angle, npy_double)
UNARY_UFUNC(sqrt, dual)
UNARY_UFUNC(cbrt, dual)
UNARY_UFUNC(square, dual)
UNARY_UFUNC(log, dual)
UNARY_UFUNC(exp, dual)
UNARY_UFUNC(sin, dual)
UNARY_UFUNC(cos, dual)
UNARY_UFUNC(tan, dual)
UNARY_UFUNC(arctan, dual)
UNARY_UFUNC(arcsin, dual)
UNARY_UFUNC(arccos, dual)
UNARY_UFUNC(sinh, dual)
UNARY_UFUNC(cosh, dual)
UNARY_UFUNC(tanh, dual)
UNARY_UFUNC(negative, dual)
UNARY_UFUNC(conjugate, dual)
UNARY_GEN_UFUNC(reciprocal, inverse, dual)
UNARY_GEN_UFUNC(invert, inverse, dual)
UNARY_UFUNC(normalized, dual)

static void
dual_positive_ufunc(char** args, npy_intp* dimensions, npy_intp* steps, void* NPY_UNUSED(data)) {
  char *ip1 = args[0], *op1 = args[1];
  npy_intp is1 = steps[0], os1 = steps[1];
  npy_intp n = dimensions[0];
  npy_intp i;
  for(i = 0; i < n; i++, ip1 += is1, op1 += os1) {
    const dual in1 = *(dual *)ip1;
    *((dual *)op1) = in1;
  }
}

// This is a macro that will be used to define the various basic binary
// dual functions, so that they can be applied quickly to a
// numpy array of duals.
#define BINARY_GEN_UFUNC(ufunc_name, func_name, arg_type1, arg_type2, ret_type) \
  static void                                                           \
  dual_##ufunc_name##_ufunc(char** args, npy_intp* dimensions,    \
                                  npy_intp* steps, void* NPY_UNUSED(data)) { \
    /* fprintf (stderr, "file %s, line %d, dual_%s_ufunc.\n", __FILE__, __LINE__, #ufunc_name); */ \
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];                \
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];            \
    npy_intp n = dimensions[0];                                         \
    npy_intp i;                                                         \
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1) {        \
      const arg_type1 in1 = *(arg_type1 *)ip1;                          \
      const arg_type2 in2 = *(arg_type2 *)ip2;                          \
      *((ret_type *)op1) = dual_##func_name(in1, in2);            \
    };                                                                  \
  };
// A couple special-case versions of the above
#define BINARY_UFUNC(name, ret_type)                    \
  BINARY_GEN_UFUNC(name, name, dual, dual, ret_type)
#define BINARY_SCALAR_UFUNC(name, ret_type)                             \
  BINARY_GEN_UFUNC(name##_scalar, name##_scalar, dual, npy_double, ret_type) \
  BINARY_GEN_UFUNC(scalar_##name, scalar_##name, npy_double, dual, ret_type)
// And these all do the work mentioned above, using the macros
BINARY_UFUNC(add, dual)
BINARY_UFUNC(subtract, dual)
BINARY_UFUNC(multiply, dual)
BINARY_UFUNC(divide, dual)
BINARY_GEN_UFUNC(true_divide, divide, dual, dual, dual)
BINARY_GEN_UFUNC(floor_divide, divide, dual, dual, dual)
BINARY_UFUNC(power, dual)
BINARY_UFUNC(copysign, dual)

BINARY_UFUNC(equal, npy_bool)
BINARY_UFUNC(not_equal, npy_bool)
BINARY_UFUNC(less, npy_bool)
BINARY_UFUNC(less_equal, npy_bool)


BINARY_SCALAR_UFUNC(add, dual)
BINARY_SCALAR_UFUNC(subtract, dual)
BINARY_SCALAR_UFUNC(multiply, dual)
BINARY_SCALAR_UFUNC(divide, dual)
BINARY_GEN_UFUNC(true_divide_scalar, divide_scalar, dual, npy_double, dual)
BINARY_GEN_UFUNC(floor_divide_scalar, divide_scalar, dual, npy_double, dual)
BINARY_GEN_UFUNC(scalar_true_divide, scalar_divide, npy_double, dual, dual)
BINARY_GEN_UFUNC(scalar_floor_divide, scalar_divide, npy_double, dual, dual)
BINARY_SCALAR_UFUNC(power, dual)




// This contains assorted other top-level methods for the module
/*** QUAT_DBL
static PyMethodDef QuaternionMethods[] = {
  {"slerp_evaluate", pyquaternion_slerp_evaluate, METH_VARARGS,
   "Interpolate linearly along the geodesic between two rotors \n\n"
   "See also `numpy.slerp_vectorized` for a vectorized version of this function, and\n"
   "`quaternion.slerp` for the most useful form, which automatically finds the correct\n"
   "rotors to interpolate and the relative time to which they must be interpolated."},
  {"squad_evaluate", pyquaternion_squad_evaluate, METH_VARARGS,
   "Interpolate linearly along the geodesic between two rotors\n\n"
   "See also `numpy.squad_vectorized` for a vectorized version of this function, and\n"
   "`quaternion.squad` for the most useful form, which automatically finds the correct\n"
   "rotors to interpolate and the relative time to which they must be interpolated."},
  {NULL, NULL, 0, NULL}
};
***/

int dual_elsize = sizeof(dual);

typedef struct { char c; dual q; } align_test;
int dual_alignment = offsetof(align_test, q);


/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
//                                                             //
//  Everything above was preparation for the following set up  //
//                                                             //
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////


#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "numpy_dual",
    NULL,
    -1,
    NULL, /*** QUAT_DBL DualMethods,***/
    NULL,
    NULL,
    NULL,
    NULL
};

#define INITERROR return NULL

// This is the initialization function that does the setup
PyMODINIT_FUNC PyInit__dual_number(void) {

#else

#define INITERROR return

// This is the initialization function that does the setup
PyMODINIT_FUNC init_dual_number(void) {

#endif

  PyObject *module;
  PyObject *tmp_ufunc;
  int dualNum;
  int arg_types[3];
  PyArray_Descr* arg_dtypes[6];
  PyObject* numpy;
  PyObject* numpy_dict;

  // Initialize a (for now, empty) module
#if PY_MAJOR_VERSION >= 3
  module = PyModule_Create(&moduledef);
#else
  module = Py_InitModule("numpy_dual", DualMethods);
#endif

  if(module==NULL) {
    INITERROR;
  }

  // Initialize numpy
  import_array();
  if (PyErr_Occurred()) {
    INITERROR;
  }
  import_umath();
  if (PyErr_Occurred()) {
    INITERROR;
  }
  numpy = PyImport_ImportModule("numpy");
  if (!numpy) {
    INITERROR;
  }
  numpy_dict = PyModule_GetDict(numpy);
  if (!numpy_dict) {
    INITERROR;
  }

  // Register the dual array base type.  Couldn't do this until
  // after we imported numpy (above)
  PyDual_Type.tp_base = &PyGenericArrType_Type;
  if (PyType_Ready(&PyDual_Type) < 0) {
    PyErr_Print();
    PyErr_SetString(PyExc_SystemError, "Could not initialize PyDual_Type.");
    INITERROR;
  }

  // The array functions, to be used below.  This InitArrFuncs
  // function is a convenient way to set all the fields to zero
  // initially, so we don't get undefined behavior.
  PyArray_InitArrFuncs(&_PyDual_ArrFuncs);
  _PyDual_ArrFuncs.nonzero = (PyArray_NonzeroFunc*)QUATERNION_nonzero;
  _PyDual_ArrFuncs.copyswap = (PyArray_CopySwapFunc*)QUATERNION_copyswap;
  _PyDual_ArrFuncs.copyswapn = (PyArray_CopySwapNFunc*)QUATERNION_copyswapn;
  _PyDual_ArrFuncs.setitem = (PyArray_SetItemFunc*)QUATERNION_setitem;
  _PyDual_ArrFuncs.getitem = (PyArray_GetItemFunc*)QUATERNION_getitem;
  _PyDual_ArrFuncs.compare = NULL; /*** DUAL_DBL (PyArray_CompareFunc*)QUATERNION_compare; ***/
  _PyDual_ArrFuncs.argmax = (PyArray_ArgFunc*)QUATERNION_argmax;
  _PyDual_ArrFuncs.fillwithscalar = (PyArray_FillWithScalarFunc*)QUATERNION_fillwithscalar;

  // The dual array descr
  dual_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
  dual_descr->typeobj = &PyDual_Type;
  dual_descr->kind = 'V';
  dual_descr->type = 'q';
  dual_descr->byteorder = '=';
  dual_descr->flags = NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM;
  dual_descr->type_num = 0; // assigned at registration
  dual_descr->elsize = dual_elsize;
  dual_descr->alignment = dual_alignment;
  dual_descr->subarray = NULL;
  dual_descr->fields = NULL;
  dual_descr->names = NULL;
  dual_descr->f = &_PyDual_ArrFuncs;
  dual_descr->metadata = NULL;
  dual_descr->c_metadata = NULL;

  Py_INCREF(&PyDual_Type);
  dualNum = PyArray_RegisterDataType(dual_descr);

  if (dualNum < 0) {
    INITERROR;
  }

  register_cast_function(NPY_BOOL, dualNum, (PyArray_VectorUnaryFunc*)BOOL_to_dual);
  register_cast_function(NPY_BYTE, dualNum, (PyArray_VectorUnaryFunc*)BYTE_to_dual);
  register_cast_function(NPY_UBYTE, dualNum, (PyArray_VectorUnaryFunc*)UBYTE_to_dual);
  register_cast_function(NPY_SHORT, dualNum, (PyArray_VectorUnaryFunc*)SHORT_to_dual);
  register_cast_function(NPY_USHORT, dualNum, (PyArray_VectorUnaryFunc*)USHORT_to_dual);
  register_cast_function(NPY_INT, dualNum, (PyArray_VectorUnaryFunc*)INT_to_dual);
  register_cast_function(NPY_UINT, dualNum, (PyArray_VectorUnaryFunc*)UINT_to_dual);
  register_cast_function(NPY_LONG, dualNum, (PyArray_VectorUnaryFunc*)LONG_to_dual);
  register_cast_function(NPY_ULONG, dualNum, (PyArray_VectorUnaryFunc*)ULONG_to_dual);
  register_cast_function(NPY_LONGLONG, dualNum, (PyArray_VectorUnaryFunc*)LONGLONG_to_dual);
  register_cast_function(NPY_ULONGLONG, dualNum, (PyArray_VectorUnaryFunc*)ULONGLONG_to_dual);
  register_cast_function(NPY_FLOAT, dualNum, (PyArray_VectorUnaryFunc*)FLOAT_to_dual);
  register_cast_function(NPY_DOUBLE, dualNum, (PyArray_VectorUnaryFunc*)DOUBLE_to_dual);
  register_cast_function(NPY_LONGDOUBLE, dualNum, (PyArray_VectorUnaryFunc*)LONGDOUBLE_to_dual);

  // These macros will be used below
  #define REGISTER_UFUNC(name)                                          \
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name), \
                                dual_descr->type_num, dual_##name##_ufunc, arg_types, NULL)
  #define REGISTER_SCALAR_UFUNC(name)                                   \
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name), \
                                dual_descr->type_num, dual_scalar_##name##_ufunc, arg_types, NULL)
  #define REGISTER_UFUNC_SCALAR(name)                                   \
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name), \
                                dual_descr->type_num, dual_##name##_scalar_ufunc, arg_types, NULL)
  #define REGISTER_NEW_UFUNC_GENERAL(pyname, cname, nargin, nargout, doc) \
    tmp_ufunc = PyUFunc_FromFuncAndData(NULL, NULL, NULL, 0, nargin, nargout, \
                                        PyUFunc_None, #pyname, doc, 0); \
    PyUFunc_RegisterLoopForType((PyUFuncObject *)tmp_ufunc,             \
                                dual_descr->type_num, dual_##cname##_ufunc, arg_types, NULL); \
    PyDict_SetItemString(numpy_dict, #pyname, tmp_ufunc);               \
    Py_DECREF(tmp_ufunc)
  #define REGISTER_NEW_UFUNC(name, nargin, nargout, doc)                \
    REGISTER_NEW_UFUNC_GENERAL(name, name, nargin, nargout, doc)

  // quat -> bool
  arg_types[0] = dual_descr->type_num;
  arg_types[1] = NPY_BOOL;
  REGISTER_UFUNC(isnan);
  /* // Already works: REGISTER_UFUNC(nonzero); */
  REGISTER_UFUNC(isinf);
  REGISTER_UFUNC(isfinite);

  // quat -> double
  arg_types[0] = dual_descr->type_num;
  arg_types[1] = NPY_DOUBLE;
  REGISTER_NEW_UFUNC(norm, 1, 1,
                     "Return Cayley norm (square of the absolute value) of each dual.\n");
  REGISTER_UFUNC(absolute);
  REGISTER_NEW_UFUNC_GENERAL(angle_of_rotor, angle, 1, 1,
                             "Return angle of rotation, assuming input is a unit rotor\n");

  // quat -> quat
  arg_types[0] = dual_descr->type_num;
  arg_types[1] = dual_descr->type_num;
  REGISTER_UFUNC(sqrt);
  REGISTER_UFUNC(cbrt);
  REGISTER_UFUNC(square);
  REGISTER_UFUNC(log);
  REGISTER_UFUNC(exp);
  REGISTER_UFUNC(sin);
  REGISTER_UFUNC(cos);
  REGISTER_UFUNC(tan);
  REGISTER_UFUNC(arctan);
  REGISTER_UFUNC(arcsin);
  REGISTER_UFUNC(arccos);
  REGISTER_UFUNC(cosh);
  REGISTER_UFUNC(sinh);
  REGISTER_UFUNC(tanh);  
  REGISTER_UFUNC(negative);
  REGISTER_UFUNC(positive);
  REGISTER_UFUNC(conjugate);
  REGISTER_UFUNC(invert);
  REGISTER_UFUNC(reciprocal);
  REGISTER_NEW_UFUNC(normalized, 1, 1,
                     "Normalize all duals in this array\n");

  // quat, quat -> bool
  arg_types[0] = dual_descr->type_num;
  arg_types[1] = dual_descr->type_num;
  arg_types[2] = NPY_BOOL;
  REGISTER_UFUNC(equal);
  REGISTER_UFUNC(not_equal);
  REGISTER_UFUNC(less);
  REGISTER_UFUNC(less_equal);


  // quat, quat -> quat
  arg_types[0] = dual_descr->type_num;
  arg_types[1] = dual_descr->type_num;
  arg_types[2] = dual_descr->type_num;
  REGISTER_UFUNC(add);
  REGISTER_UFUNC(subtract);
  REGISTER_UFUNC(multiply);
  REGISTER_UFUNC(divide);
  REGISTER_UFUNC(true_divide);
  REGISTER_UFUNC(floor_divide);
  REGISTER_UFUNC(power);
  REGISTER_UFUNC(copysign);

  // double, quat -> quat
  arg_types[0] = NPY_DOUBLE;
  arg_types[1] = dual_descr->type_num;
  arg_types[2] = dual_descr->type_num;
  REGISTER_SCALAR_UFUNC(add);
  REGISTER_SCALAR_UFUNC(subtract);
  REGISTER_SCALAR_UFUNC(multiply);
  REGISTER_SCALAR_UFUNC(divide);
  REGISTER_SCALAR_UFUNC(true_divide);
  REGISTER_SCALAR_UFUNC(floor_divide);
  REGISTER_SCALAR_UFUNC(power);

  // quat, double -> quat
  arg_types[0] = dual_descr->type_num;
  arg_types[1] = NPY_DOUBLE;
  arg_types[2] = dual_descr->type_num;
  REGISTER_UFUNC_SCALAR(add);
  REGISTER_UFUNC_SCALAR(subtract);
  REGISTER_UFUNC_SCALAR(divide);
  REGISTER_UFUNC_SCALAR(true_divide);
  REGISTER_UFUNC_SCALAR(floor_divide);
  REGISTER_UFUNC_SCALAR(power);

/*** QUAT_DBL
  // quat, quat -> double
  arg_types[0] = dual_descr->type_num;
  arg_types[1] = quaternion_descr->type_num;
  arg_types[2] = NPY_DOUBLE;
  REGISTER_NEW_UFUNC(rotor_intrinsic_distance, 2, 1,
                     "Distance measure intrinsic to rotor manifold");
  REGISTER_NEW_UFUNC(rotor_chordal_distance, 2, 1,
                     "Distance measure from embedding of rotor manifold");
  REGISTER_NEW_UFUNC(rotation_intrinsic_distance, 2, 1,
                     "Distance measure intrinsic to rotation manifold");
  REGISTER_NEW_UFUNC(rotation_chordal_distance, 2, 1,
                     "Distance measure from embedding of rotation manifold");
***/

  /* I think before I do the following, I'll have to update numpy_dict
   * somehow, presumably with something related to
   * `PyUFunc_RegisterLoopForType`.  I should also do this for the
   * various other methods defined above. */

/*** QUAT_DBL
  // Create a custom ufunc and register it for loops.  The method for
  // doing this was pieced together from examples given on the page
  // <https://docs.scipy.org/doc/numpy/user/c-info.ufunc-tutorial.html>
  arg_dtypes[0] = PyArray_DescrFromType(NPY_DOUBLE);
  arg_dtypes[1] = quaternion_descr;
  arg_dtypes[2] = quaternion_descr;
  arg_dtypes[3] = quaternion_descr;
  arg_dtypes[4] = quaternion_descr;
  arg_dtypes[5] = quaternion_descr;
  squad_evaluate_ufunc = PyUFunc_FromFuncAndData(NULL, NULL, NULL, 0, 5, 1,
                                                 PyUFunc_None, "squad_vectorized",
                                                 "Calculate squad from arrays of (tau, q_i, a_i, b_ip1, q_ip1)\n\n"
                                                 "See `quaternion.squad` for an easier-to-use version of this function",
                                                  0);
  PyUFunc_RegisterLoopForDescr((PyUFuncObject*)squad_evaluate_ufunc,
                               quaternion_descr,
                               &squad_loop,
                               arg_dtypes,
                               NULL);
  PyDict_SetItemString(numpy_dict, "squad_vectorized", squad_evaluate_ufunc);
  Py_DECREF(squad_evaluate_ufunc);

  // Create a custom ufunc and register it for loops.  The method for
  // doing this was pieced together from examples given on the page
  // <https://docs.scipy.org/doc/numpy/user/c-info.ufunc-tutorial.html>
  arg_dtypes[0] = quaternion_descr;
  arg_dtypes[1] = quaternion_descr;
  arg_dtypes[2] = PyArray_DescrFromType(NPY_DOUBLE);
  slerp_evaluate_ufunc = PyUFunc_FromFuncAndData(NULL, NULL, NULL, 0, 3, 1,
                                                 PyUFunc_None, "slerp_vectorized",
                                                 "Calculate slerp from arrays of (q_1, q_2, tau)\n\n"
                                                 "See `quaternion.slerp` for an easier-to-use version of this function",
                                                  0);
  PyUFunc_RegisterLoopForDescr((PyUFuncObject*)slerp_evaluate_ufunc,
                               quaternion_descr,
                               &slerp_loop,
                               arg_dtypes,
                               NULL);
  PyDict_SetItemString(numpy_dict, "slerp_vectorized", slerp_evaluate_ufunc);
  Py_DECREF(slerp_evaluate_ufunc);

***/


  // Add the constant `_QUATERNION_EPS` to the module as `quaternion._eps`
/*** DUAL_DBL  PyModule_AddObject(module, "_eps", PyFloat_FromDouble(_QUATERNION_EPS));
***/ 
 
  // Finally, add this quaternion object to the quaternion module itself
  PyModule_AddObject(module, "dual_number", (PyObject *)&PyDual_Type);


#if PY_MAJOR_VERSION >= 3
    return module;
#else
    return;
#endif
}
