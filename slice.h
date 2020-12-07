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

struct Slice {
	int slice_no;
    uint8_t *luma_matrix;
    uint8_t *chroma_matrix;
    uint8_t qscale;
    uint32_t slice_size_in_mb;
    uint32_t horizontal;
    uint32_t vertical;
    uint16_t *y_data;//original data
    uint16_t *cb_data;//original data
    uint16_t *cr_data;//original data
    uint32_t mb_x;
    uint32_t mb_y;
    bool format_444;
	bool end;
	struct bitstream *bitstream;
	int16_t *working_buffer;//working buffer 

};

uint16_t encode_slice(struct Slice *param);

#endif
