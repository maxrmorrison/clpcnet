#ifndef COMMON_H
#define COMMON_H


#include <math.h>
#include <stdlib.h>
#include <string.h>


float lpc_from_cepstrum(float *lpc, const float *cepstrum);

#define LOG256 5.5451774445f
static inline float log2_approx(float x)
{
   int integer;
   float frac;
   union {
      float f;
      int i;
   } in;
   in.f = x;
   integer = (in.i>>23)-127;
   in.i -= integer<<23;
   frac = in.f - 1.5f;
   frac = -0.41445418f + frac*(0.95909232f
          + frac*(-0.33951290f + frac*0.16541097f));
   return 1+integer+frac;
}

#define log_approx(x) (0.69315f*log2_approx(x))

/** Copy n elements from src to dst. The 0* term provides compile-time type checking  */
#ifndef OVERRIDE_RNN_COPY
#define RNN_COPY(dst, src, n) (memcpy((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))
#endif

/** Copy n elements from src to dst, allowing overlapping regions. The 0* term
    provides compile-time type checking */
#ifndef OVERRIDE_RNN_MOVE
#define RNN_MOVE(dst, src, n) (memmove((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))
#endif

/** Set n elements of dst to zero */
#ifndef OVERRIDE_RNN_CLEAR
#define RNN_CLEAR(dst, n) (memset((dst), 0, (n)*sizeof(*(dst))))
#endif


#endif
