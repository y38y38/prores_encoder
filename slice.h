/**
 *
 * Copyright (c) 2020 Yuusuke Miyazaki
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 *
 **/

#ifndef __SLICE_H__
#define __SLICE_H__
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

#include "prores.h"
#include "encoder.h"


struct Slice_cuda {
	int slice_no;
	int slice_num_max;
    uint8_t luma_matrix[BLOCK_IN_PIXEL];
    uint8_t chroma_matrix[BLOCK_IN_PIXEL];
    uint32_t slice_size_in_mb;
    uint32_t horizontal;
    uint32_t vertical;
    bool format_444;
	bool end;
};
#ifdef CUDA_ENCODER
__global__
#endif
void encode_slice(int slice_no, struct Slice_cuda * slice_param, uint8_t *qscale_table, uint16_t *y_data, uint16_t * cb_data, uint16_t * cr_data, struct bitstream *stream, uint16_t* slice_size_table, int16_t *buffer, double* kc_value);

#endif
