// Copyright (c) 2017, Michael Boyle
// See LICENSE file for details: <https://github.com/moble/dual/blob/master/LICENSE>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
  #include "math_msvc_compatibility.h"
#else
  #include <math.h>
#endif

#include <stdio.h>
#include <float.h>

#include "dual_number.h"



dual
dual_scalar_power(double s, dual q)
{
  double ar = pow(q.re, s-1);
  dual r = { ar*s, s*ar*q.im };
  return r;
}


dual
dual_cbrt(dual q) {
  return dual_scalar_power(1./3, q);
}



/*** DUAL_DBL 
dual
dual_log(dual q)
{
  double b = sqrt(q.x*q.x + q.y*q.y + q.z*q.z);
  if(fabs(b) <= _dual_EPS*fabs(q.w)) {
    if(q.w<0.0) {
      // fprintf(stderr, "Input dual(%.15g, %.15g, %.15g, %.15g) has no unique logarithm; returning one arbitrarily.", q.w, q.x, q.y, q.z);
      if(fabs(q.w+1)>_dual_EPS) {
        dual r = {log(-q.w), M_PI, 0., 0.};
        return r;
      } else {
        dual r = {0., M_PI, 0., 0.};
        return r;
      }
    } else {
      dual r = {log(q.w), 0., 0., 0.};
      return r;
    }
  } else {
    double v = atan2(b, q.w);
    double f = v/b;
    dual r = { log(q.w*q.w+b*b)/2.0, f*q.x, f*q.y, f*q.z };
    return r;
  }
}

double
_dual_scalar_log(double s) { return log(s); }

***/


/* Unlike the dual^dual power, this is unambiguous. */
/*** DUAL_DBL
dual
dual_scalar_power(double s, dual q)
{
  if(s==0.0) { // log(s)=-inf 
    if(! dual_nonzero(q)) {
      dual r = {1.0, 0.0, 0.0, 0.0}; // consistent with python
      return r;
    } else {
      dual r = {0.0, 0.0, 0.0, 0.0}; // consistent with python
      return r;
    }
  } else if(s<0.0) { // log(s)=nan 
    // fprintf(stderr, "Input scalar (%.15g) has no unique logarithm; returning one arbitrarily.", s);
    dual t = {log(-s), M_PI, 0, 0};
    return dual_exp(dual_multiply(q, t));
  }
  return dual_exp(dual_multiply_scalar(q, log(s)));
}

dual
dual_exp(dual q)
{
  double vnorm = sqrt(q.x*q.x + q.y*q.y + q.z*q.z);
  if (vnorm > _dual_EPS) {
    double s = sin(vnorm) / vnorm;
    double e = exp(q.w);
    dual r = {e*cos(vnorm), e*s*q.x, e*s*q.y, e*s*q.z};
    return r;
  } else {
    dual r = {exp(q.w), 0, 0, 0};
    return r;
  }
}

***/

#ifdef __cplusplus
}
#endif
