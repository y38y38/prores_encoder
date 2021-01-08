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
	uint16_t *cr_data
};
void encode_slices2(struct Slice_cuda, param, uint16_t * slice_size_table, struct bistream *bitstream);
#endif
