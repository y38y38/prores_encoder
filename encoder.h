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

#define HORIZONTAL  (128)
#define VIRTICAL    (16)
#define VIRTICAL2   (16)

extern void encoder_init(void);
extern uint8_t *encode_frame(uint16_t *y_data, uint16_t *cb_data, uint16_t *cr_data, uint32_t *encode_frame_size);
#endif
