// Copyright (c) 2017, Michael Boyle
// See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

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

#include "quaternion.h"

quaternion
quaternion_sqrt(quaternion q)
{
  double absolute = quaternion_norm(q);  // pre-square-root
  if(absolute<=DBL_MIN) {
      quaternion r = {0.0, 0.0, 0.0, 0.0};
      return r;
  }
  absolute = sqrt(absolute);
  if(fabs(absolute+q.w)<_QUATERNION_EPS*absolute) {
    quaternion r = {0.0, sqrt(absolute), 0.0, 0.0};
    return r;
  } else {
    double c = sqrt(0.5/(absolute+q.w));
    quaternion r = {(absolute+q.w)*c, q.x*c, q.y*c, q.z*c};
    return r;
  }
}

quaternion
quaternion_log(quaternion q)
{
  double b = sqrt(q.x*q.x + q.y*q.y + q.z*q.z);
  if(fabs(b) <= _QUATERNION_EPS*fabs(q.w)) {
    if(q.w<0.0) {
      // fprintf(stderr, "Input quaternion(%.15g, %.15g, %.15g, %.15g) has no unique logarithm; returning one arbitrarily.", q.w, q.x, q.y, q.z);
      if(fabs(q.w+1)>_QUATERNION_EPS) {
        quaternion r = {log(-q.w), M_PI, 0., 0.};
        return r;
      } else {
        quaternion r = {0., M_PI, 0., 0.};
        return r;
      }
    } else {
      quaternion r = {log(q.w), 0., 0., 0.};
      return r;
    }
  } else {
    double v = atan2(b, q.w);
    double f = v/b;
    quaternion r = { log(q.w*q.w+b*b)/2.0, f*q.x, f*q.y, f*q.z };
    return r;
  }
}

double
_quaternion_scalar_log(double s) { return log(s); }

quaternion
quaternion_scalar_power(double s, quaternion q)
{
  /* Unlike the quaternion^quaternion power, this is unambiguous. */
  if(s==0.0) { /* log(s)=-inf */
    if(! quaternion_nonzero(q)) {
      quaternion r = {1.0, 0.0, 0.0, 0.0}; /* consistent with python */
      return r;
    } else {
      quaternion r = {0.0, 0.0, 0.0, 0.0}; /* consistent with python */
      return r;
    }
  } else if(s<0.0) { /* log(s)=nan */
    // fprintf(stderr, "Input scalar (%.15g) has no unique logarithm; returning one arbitrarily.", s);
    quaternion t = {log(-s), M_PI, 0, 0};
    return quaternion_exp(quaternion_multiply(q, t));
  }
  return quaternion_exp(quaternion_multiply_scalar(q, log(s)));
}

quaternion
quaternion_exp(quaternion q)
{
  double vnorm = sqrt(q.x*q.x + q.y*q.y + q.z*q.z);
  if (vnorm > _QUATERNION_EPS) {
    double s = sin(vnorm) / vnorm;
    double e = exp(q.w);
    quaternion r = {e*cos(vnorm), e*s*q.x, e*s*q.y, e*s*q.z};
    return r;
  } else {
    quaternion r = {exp(q.w), 0, 0, 0};
    return r;
  }
}

#ifdef __cplusplus
}
#endif
