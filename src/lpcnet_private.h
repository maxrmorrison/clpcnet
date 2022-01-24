#ifndef LPCNET_PRIVATE_H
#define LPCNET_PRIVATE_H

#include "celt_lpc.h"
#include "common.h"
#include "freq.h"
#include "lpcnet.h"

#define PITCH_MIN_PERIOD 32  //
#define PITCH_MAX_PERIOD 512 //

#define PITCH_FRAME_SIZE 512                                 //
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD + PITCH_FRAME_SIZE) //

#define FORBIDDEN_INTERP 7

struct LPCNetEncState
{
  float analysis_mem[OVERLAP_SIZE]; //
  int pcount;                                //
  float pitch_mem[LPC_ORDER];                //
  float pitch_filt;                          //
  float xc[10][PITCH_MAX_PERIOD + 1];        //
  float frame_weight[10];                    //
  float exc_buf[PITCH_BUF_SIZE];             //
  float pitch_max_path[2][PITCH_MAX_PERIOD]; //
  float pitch_max_path_all;                  //
  int best_i;                                //
  float lpc[LPC_ORDER];                      //
  float features[4][NB_TOTAL_FEATURES]; //
  float sig_mem[LPC_ORDER];             //
  int exc_mem;                          //
};

void process_superframe(LPCNetEncState *st, FILE *ffeat);

void compute_frame_features(LPCNetEncState *st, const float *in);

#endif
