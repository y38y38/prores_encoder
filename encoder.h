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

struct encoder_param {
    uint8_t *luma_matrix;
    uint8_t *chroma_matrix;
    uint32_t qscale_table_size;
    uint8_t *qscale_table;
    uint32_t block_num;
    uint32_t horizontal;
    uint32_t vertical;
    uint16_t *y_data;
    uint16_t *cb_data;
    uint16_t *cr_data;
};
extern void encoder_init(void);
extern uint8_t *encode_frame(struct encoder_param* param, uint32_t *encode_frame_size);
extern int32_t GetSliceNum(int32_t horizontal, int32_t vertical, int32_t sliceSize);

#endif
