/* Copyright (c) 2018 Mozilla */
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

#ifndef _LPCNET_H_
#define _LPCNET_H_


#define NB_FEATURES 38
#define NB_TOTAL_FEATURES 55


typedef struct LPCNetEncState LPCNetEncState;


/** Gets the size of an <code>LPCNetEncState</code> structure.
  * @returns The size in bytes.
  */
int lpcnet_encoder_get_size();

/** Initializes a previously allocated encoder state
  * The memory pointed to by st must be at least the size returned by lpcnet_encoder_get_size().
  * This is intended for applications which use their own allocator instead of malloc.
  * @see lpcnet_encoder_create(),lpcnet_encoder_get_size()
  * @param [in] st <tt>LPCNetEncState*</tt>: Encoder state
  * @retval 0 Success
  */
int lpcnet_encoder_init(LPCNetEncState *st);

/** Allocates and initializes an encoder state.
  *  @returns The newly created state
  */
LPCNetEncState *lpcnet_encoder_create();


#endif
