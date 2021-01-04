
/**
 *
 * Copyright (c) 2020 Yuusuke Miyazaki
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 *
 **/

#ifndef __VLC_H__
#define __VLC_H__

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>


#include "bitstream_cuda.h"

#ifdef CUDA_ENCODER
__device__
#endif
uint32_t entropy_encode_ac_coefficients(int16_t*coefficients, int32_t numBlocks, struct bitstream *bitstream);

#ifdef CUDA_ENCODER
__device__
#endif
void entropy_encode_dc_coefficients(int16_t*coefficients, int32_t numBlocks, struct bitstream *bitstream);

#endif

