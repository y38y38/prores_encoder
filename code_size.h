/**
 *
 * Copyright (c) 2020 Yuusuke Miyazaki
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 *
 **/
#ifndef _SLICE_SIZE_H_
#define _SLICE_SIZE_H_
#include <stdint.h>

struct code_size  {
    uint8_t x;
    uint8_t y;
    uint8_t slice_header_size;
    uint16_t coded_size_of_y_data;
    uint16_t coded_size_of_cb_data;
};
extern struct code_size code_sizes[];

#endif
