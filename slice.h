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

extern const uint8_t block_pattern_scan_table[64];
extern uint8_t block_pattern_scan_read_order_table[64];
extern uint32_t encode_slice(uint16_t *y_data, uint16_t *cb_data, uint16_t *cr_data, uint32_t  mb_x, uint32_t mb_y, uint8_t * luma_matrix, uint8_t *chroma_matrix);

extern uint8_t luma_matrix_[64];
extern uint8_t chroma_matrix_[64];
#endif
