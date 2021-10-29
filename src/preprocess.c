/* Copyright (c) 2017-2018 Mozilla */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "lpcnet.h"
#include "lpcnet_private.h"


/******************************************************************************
Constants
******************************************************************************/


#ifndef FRAME_SIZE
#define FRAME_SIZE 160
#endif

#ifndef HALF_FRAME
#define HALF_FRAME (FRAME_SIZE / 2)
#endif

#ifndef LOG256
#define LOG256 5.5451774445f
#endif

#ifndef LPC_ORDER
#define LPC_ORDER 16
#endif

#ifndef NB_BANDS
#define NB_BANDS 18
#endif

#ifndef OVERLAP_SIZE
#define OVERLAP_SIZE 160
#endif

#ifndef PITCH_MAX_PERIOD
#define PITCH_MAX_PERIOD 256
#endif

#ifndef WINDOW_SIZE
#define WINDOW_SIZE (FRAME_SIZE + OVERLAP_SIZE)
#endif

#ifndef FREQ_SIZE
#define FREQ_SIZE (WINDOW_SIZE / 2 + 1)
#endif

#ifndef log_approx
#define log_approx(x) (0.69315f * log2_approx(x))
#endif

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif


/******************************************************************************
Conversions
******************************************************************************/


static inline float mulaw_to_linear(float u) {
  /* Convert mulaw-encoded audio to linear */
  float s;
  float scale_1 = 32768.f / 255.f;
  u = u - 128;
  s = u >= 0 ? 1 : -1;
  u = fabs(u);
  return s * scale_1 * (exp(u / 128. * LOG256) - 1);
}


static inline int linear_to_mulaw(float x) {
  /* Convert linear audio to mulaw */
  float u;
  float scale = 255.f / 32768.f;
  int s = x >= 0 ? 1 : -1;
  x = fabs(x);
  u = (s * (128 * log_approx(1 + scale * x) / LOG256));
  u = 128 + u;
  if (u < 0)
    u = 0;
  if (u > 255)
    u = 255;
  return (int)floor(.5 + u);
}


/******************************************************************************
File output
******************************************************************************/


void write_audio(LPCNetEncState *st, const short *pcm, FILE *file) {
  // Iterate over frames in a block
  for (int k = 0; k < 4; k++) {

    // Container for sample-level features
    unsigned char data[4 * FRAME_SIZE];

    // Iterate over samples in a frame
    for (int i = 0; i < FRAME_SIZE; i++) {

      // Compute prediction from lpc coefficients and previous samples
      float p = 0;
      for (int j = 0; j < LPC_ORDER; j++)
        p -= st->features[k][2 * NB_BANDS + 3 + j] * st->sig_mem[j];

      // Compute excitation from sample and prediction
      float e = linear_to_mulaw(pcm[k * FRAME_SIZE + i] - p);

      // Mu-law encoded signal
      data[4 * i] = linear_to_mulaw(st->sig_mem[0]);

      // Mu-law encoded prediction
      data[4 * i + 1] = linear_to_mulaw(p);

      // Input excitation
      data[4 * i + 2] = st->exc_mem;

      // Target excitation
      data[4 * i + 3] = e;

      // Delay signal by one
      unsigned int size = (LPC_ORDER - 1) * sizeof(float);
      memmove(&st->sig_mem[1], &st->sig_mem[0], size);

      // Bound excitation
      e = min(255, max(0, e));

      // Store computed values for next iteration
      st->sig_mem[0] = p + mulaw_to_linear(e);
      st->exc_mem = e;
    }

    // Write sample-rate features to disk
    fwrite(data, 4 * FRAME_SIZE, 1, file);
  }
}


/******************************************************************************
Entry point
******************************************************************************/


int main(int argc, char **argv) {
  float x[FRAME_SIZE];
  FILE *output_sample_file = NULL;
  short pcm[FRAME_SIZE] = {0};
  short pcmbuf[FRAME_SIZE * 4] = {0};

  // Create encoder
  LPCNetEncState *st = lpcnet_encoder_create();

  // Open input audio file
  FILE *input_file = fopen(argv[1], "rb");

  // Open output feature file
  FILE *output_feature_file = fopen(argv[2], "wb");

  // Open output file for sample-rate features and training targets
  if (argc == 4) output_sample_file = fopen(argv[3], "wb");

  // Read in up to FRAME_SIZE samples
  while (fread(pcm, sizeof(short), FRAME_SIZE, input_file) == FRAME_SIZE) {

    // Cast to float
    for (int i = 0; i < FRAME_SIZE; i++) x[i] = pcm[i];

    // Compute pitch, correlation, and bark-scale coefficients
    compute_frame_features(st, x);

    // Copy frame into position in 4-frame buffer
    memcpy(&pcmbuf[st->pcount * FRAME_SIZE], pcm, FRAME_SIZE * sizeof(*pcm));
    st->pcount++;

    // Running on groups of 4 frames
    if (st->pcount == 4) {
      process_superframe(st, output_feature_file);

      // If training, write audio frame
      if (output_sample_file) write_audio(st, pcmbuf, output_sample_file);

      // Reset count
      st->pcount = 0;
    }
  }

  // Close files
  fclose(input_file);
  fclose(output_feature_file);
  if (output_sample_file) fclose(output_sample_file);

  // Clean-up encoder memory
  free(st);

  return 0;
}
