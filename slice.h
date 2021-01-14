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
    uint8_t *luma_matrix;
    uint8_t *chroma_matrix;
    uint32_t slice_size_in_mb;
    uint32_t horizontal;
    uint32_t vertical;
    bool format_444;
	bool end;
	uint8_t *qscale_table;
	uint16_t *y_data;
	uint16_t *cb_data;
	uint16_t *cr_data;
};
void encode_slices2(struct Slice_cuda* param, int slice_no, uint16_t * slice_size_table, struct bitstream *bitstream, int16_t*working_buffer, double *kc_value);

int mbXFormSliceNo(struct Slice_cuda* slice_param, int slice_no);
int mbYFormSliceNo(struct Slice_cuda* slice_param, int slice_no);

void getYver2(uint16_t *out, uint16_t *in, uint32_t mb_x, uint32_t mb_y, int32_t mb_size, int32_t horizontal, int32_t vertical);
void getCver2(uint16_t *out, uint16_t *in, uint32_t mb_x, uint32_t mb_y, int32_t mb_size, int32_t horizontal, int32_t vertical);

void dct_and_quant(int16_t *pixel, uint8_t *matrix, int slice_size_in_mb, int mb_in_block, double *kc_value, uint8_t qscale);

#endif
