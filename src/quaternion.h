// Copyright (c) 2017, Michael Boyle
// See LICENSE file for details: <https://github.com/moble/dual/blob/master/LICENSE>

#ifndef __dual_H__
#define __dual_H__

#ifdef __cplusplus
extern "C" {
#endif

  #if defined(_MSC_VER)
    #include "math_msvc_compatibility.h"
  #else
    #include <math.h>
  #endif

  #define _dual_EPS 1e-14

  #if defined(_MSC_VER)
    #define NPY_INLINE __inline
  #elif defined(__GNUC__)
    #if defined(__STRICT_ANSI__)
      #define NPY_INLINE __inline__
    #else
      #define NPY_INLINE inline
    #endif
  #else
    #define NPY_INLINE
  #endif

  typedef struct {
    double re;
    double im;
  } dual;


  // Unary bool returners
  static NPY_INLINE int dual_isnan(dual q) {
    return isnan(q.re) || isnan(q.im);
  }
  static NPY_INLINE int dual_nonzero(dual q) {
    if(dual_isnan(q)) { return 1; }
    return ! (q.re == 0 && q.im == 0);
  }
  static NPY_INLINE int dual_isinf(dual q) {
    return isinf(q.re) || isinf(q.im);
  }
  static NPY_INLINE int dual_isfinite(dual q) {
    return isfinite(q.re) && isfinite(q.im);
  }


  // Binary bool returners
  static NPY_INLINE int dual_equal(dual q1, dual q2) {
    return
      !dual_isnan(q1) &&
      !dual_isnan(q2) &&
      q1.re == q2.re &&
      q1.im == q2.im;
  }
  static NPY_INLINE int dual_not_equal(dual q1, dual q2) {
    return !dual_equal(q1, q2);
  }
  static NPY_INLINE int dual_less(dual q1, dual q2) {
    return
      (!dual_isnan(q1) && !dual_isnan(q2))
      &&
      (q1.re != q2.re ? q1.re < q2.re :
       q1.im != q2.im ? q1.im < q2.im : 0);
  }
  static NPY_INLINE int dual_greater(dual q1, dual q2) {
    return
      (!dual_isnan(q1) && !dual_isnan(q2))
      &&
      (q1.re != q2.re ? q1.re > q2.re :
       q1.im != q2.im ? q1.im > q2.im : 0);
  }
  static NPY_INLINE int dual_less_equal(dual q1, dual q2) {
    return
      (!dual_isnan(q1) && !dual_isnan(q2))
      &&
      (q1.re != q2.re ? q1.re < q2.re :
       q1.im != q2.im ? q1.im < q2.im : 1);
    // Note that the final possibility is 1, whereas in
    // `dual_less` it was 0.  This distinction correctly
    // accounts for equality.
  }
  static NPY_INLINE int dual_greater_equal(dual q1, dual q2) {
    return
      (!dual_isnan(q1) && !dual_isnan(q2))
      &&
      (q1.re != q2.re ? q1.re > q2.re :
       q1.im != q2.im ? q1.im > q2.im : 1);
    // Note that the final possibility is 1, whereas in
    // `dual_greater` it was 0.  This distinction correctly
    // accounts for equality.
  }


  // Unary float returners
  dual dual_log(dual q); // Pre-declare; declared again below, in its rightful place
  
  /*** XXX: stub; is likely not the right definition ***/
  static NPY_INLINE double dual_norm(dual q) {
    return q.re*q.re + q.im*q.im;
  }
  static NPY_INLINE double dual_absolute(dual q) {
    return sqrt(q.re*q.re + q.im*q.im);
  }


  static NPY_INLINE double dual_angle(dual q) {
    return M_PI; //2 * dual_absolute( dual_log( q ) );    /*** DUAL_DBL **///
  }


  // Unary dual returners
  dual dual_sqrt(dual q);
  dual dual_log(dual q);
  dual dual_exp(dual q);

  /*** XXX: stub; should probably be removed */
  static NPY_INLINE dual dual_normalized(dual q) {
    double q_abs = dual_absolute(q);
    dual r = {q.re/q_abs, q.im/q_abs};
    return r;
  }

  static NPY_INLINE dual dual_negative(dual q) {
    dual r = {-q.re, -q.im};
    return r;
  }
  static NPY_INLINE dual dual_conjugate(dual q) {
    dual r = {q.re, -q.im};
    return r;
  }
  static NPY_INLINE dual dual_inverse(dual q) {
    double norm = dual_norm(q);
    dual r = {q.re/norm, -q.im/norm};
    return r;
  }

  // dual-dual binary dual returners
  static NPY_INLINE dual dual_copysign(dual q1, dual q2) {
    dual r = {
      copysign(q1.re, q2.re),
      copysign(q1.im, q2.im),
    };
    return r;
  }


  // dual-dual/dual-scalar binary dual returners
  static NPY_INLINE dual dual_add(dual q1, dual q2) {
    dual r = {
      q1.re + q2.re,
      q1.im + q2.im,
    };
    return r;
  }
  static NPY_INLINE void dual_inplace_add(dual* q1, dual q2) {
    q1->re += q2.re;
    q1->im += q2.im;
    return;
  }
  static NPY_INLINE dual dual_scalar_add(double s, dual q) {
    dual r = {s + q.re, q.im};
    return r;
  }
  static NPY_INLINE void dual_inplace_scalar_add(double s, dual* q) {
    q->re += s;
    return;
  }
  static NPY_INLINE dual dual_add_scalar(dual q, double s) {
    dual r = {s + q.re, q.im};
    return r;
  }
  static NPY_INLINE void dual_inplace_add_scalar(dual* q, double s) {
    q->re += s;
    return;
  }
  static NPY_INLINE dual dual_subtract(dual q1, dual q2) {
    dual r = {
      q1.re - q2.re,
      q1.im - q2.im,
    };
    return r;
  }
  static NPY_INLINE void dual_inplace_subtract(dual* q1, dual q2) {
    q1->re -= q2.re;
    q1->im -= q2.im;
    return;
  }
  static NPY_INLINE dual dual_scalar_subtract(double s, dual q) {
    dual r = {s - q.re, -q.im};
    return r;
  }
  static NPY_INLINE dual dual_subtract_scalar(dual q, double s) {
    dual r = {q.re - s, q.im};
    return r;
  }
  static NPY_INLINE void dual_inplace_subtract_scalar(dual* q, double s) {
    q->re -= s;
    return;
  }
  static NPY_INLINE dual dual_multiply(dual q1, dual q2) {
    dual r = {
      q1.re*q2.re,
      q1.re*q2.im + q1.im*q2.re,
    };
    return r;
  }
  static NPY_INLINE dual dual_square(dual q) {
    return dual_multiply(q, q);
  }
  static NPY_INLINE void dual_inplace_multiply(dual* q1a, dual q2) {
    dual q1 = {q1a->re, q1a->im};
    q1a->re = q1.re*q2.re;
    q1a->im = q1.re*q2.im + q1.im*q2.re;
    return;
  }
  static NPY_INLINE dual dual_scalar_multiply(double s, dual q) {
    dual r = {s*q.re, s*q.im};
    return r;
  }
  static NPY_INLINE void dual_inplace_scalar_multiply(double s, dual* q) {
    q->re *= s;
    q->im *= s;
    return;
  }
  static NPY_INLINE dual dual_multiply_scalar(dual q, double s) {
    dual r = {s*q.re, s*q.im};
    return r;
  }
  static NPY_INLINE void dual_inplace_multiply_scalar(dual* q, double s) {
    q->re *= s;
    q->im *= s;
    return;
  }
  static NPY_INLINE dual dual_divide(dual q1, dual q2) {
    dual r = {
        q1.re / q2.re,
        (q1.im*q2.re - q2.im*q1.re) / q2.re / q2.re,
    };
    return r;
  }
  static NPY_INLINE void dual_inplace_divide(dual* q1a, dual q2) {
    dual q1 = *q1a;
    q1a->re = q1.re / q2.re; 
    q1a->im = (q1.im*q2.re - q2.im*q1.re) / q2.re / q2.re;
    return;
  }
  static NPY_INLINE dual dual_scalar_divide(double s, dual q) {
    dual r = {
       s / q.re,
      -s *q.im / q.re / q.re,
    };
    return r;
  }
  /* The following function is impossible, but listed for completeness: */
  /* static NPY_INLINE void dual_inplace_scalar_divide(double* sa, dual q2) { } */
  static NPY_INLINE dual dual_divide_scalar(dual q, double s) {
    dual r = {q.re/s, q.im/s};
    return r;
  }
  static NPY_INLINE void dual_inplace_divide_scalar(dual* q, double s) {
    q->re /= s;
    q->im /= s;
    return;
  }

  /*** DUAL_DBL: XXX: how to just not define dual^dual ***/
  static NPY_INLINE dual dual_power(dual q, dual p) {
    /* Note that the following is just my chosen definition of the power. */
    /* Other definitions may disagree due to non-commutativity. */
    if(! dual_nonzero(q)) { /* log(q)=-inf */
      if(! dual_nonzero(p)) {
        dual r = {1.0, 0.0}; /* consistent with python */
        return r;
      } else {
        dual r = {0.0, 0.0}; /* consistent with python */
        return r;
      }
    }
    return dual_exp(dual_multiply(dual_log(q), p));
  }
  static NPY_INLINE void dual_inplace_power(dual* q1, dual q2) {
    /* Not overly useful as an in-place operator, but here for completeness. */
    dual q3 = dual_power(*q1,q2);
    *q1 = q3;
    return;
  }
  dual dual_scalar_power(double s, dual q);
  static NPY_INLINE void dual_inplace_scalar_power(double s, dual* q) {
    /* Not overly useful as an in-place operator, but here for completeness. */
    dual q2 = dual_scalar_power(s, *q);
    *q = q2;
    return;
  }
  static NPY_INLINE dual dual_power_scalar(dual q, double s) {
    /* Unlike the dual^dual power, this is unambiguous. */
    if(! dual_nonzero(q)) { /* log(q)=-inf */
      if(s==0) {
        dual r = {1.0, 0.0}; /* consistent with python */
        return r;
      } else {
        dual r = {0.0, 0.0}; /* consistent with python */
        return r;
      }
    }
    double as = pow(q.re, s-1);
    dual r = { as*s, s*as*q.im };
    return r;
  }
  static NPY_INLINE void dual_inplace_power_scalar(dual* q, double s) {
    /* Not overly useful as an in-place operator, but here for completeness. */
    dual q2 = dual_power_scalar(*q, s);
    *q = q2;
    return;
  }




#ifdef __cplusplus
}
#endif

#endif // __dual_H__
