/**
 *
 * Copyright (c) 2020 Yuusuke Miyazaki
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 *
 **/
#ifndef __ENCODER_H__
#define __ENCODER_H__

#include <stdint.h>

#if 0
#define MATRIX_ROW_NUM  (8)
#define MATRIX_COLUMN_NUM  (8)
#define MATRIX_NUM (MATRIX_ROW_NUM*MATRIX_COLUMN_NUM)
#endif

//max 8192 x 4096
#define MAX_SLICE_NUM	(131072)



struct encoder_param {
    uint8_t *luma_matrix;
    uint8_t *chroma_matrix;
    uint32_t qscale_table_size;
    uint8_t *qscale_table;
    uint32_t slice_size_in_mb;
    uint32_t horizontal;
    uint32_t vertical;
    uint16_t *y_data;
    uint16_t *cb_data;
    uint16_t *cr_data;
    bool format_444;
};

int32_t GetSliceNum(int32_t horizontal, int32_t vertical, int32_t sliceSize);
uint32_t GetEncodeHorizontal(int32_t horizontal);
uint32_t GetEncodeVertical(int32_t vertical);

void encoder_init(void);
uint8_t *encode_frame(struct encoder_param* param, uint32_t *encode_frame_size);

#endif
