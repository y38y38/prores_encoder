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

extern void set_frame_header(void);
extern void set_picture_header(void);
extern void encode_slices(uint16_t *y_data, uint16_t *cb_data, uint16_t *cr_data);
extern uint32_t picture_size_offset_;
extern const uint8_t block_pattern_scan_table[64];
extern uint8_t block_pattern_scan_read_order_table[64];
#endif
