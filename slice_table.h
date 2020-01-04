/**
 *
 * Copyright (c) 2020 Yuusuke Miyazaki
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 *
 **/
#ifndef _SLICE_TABLE_H_
#define _SLICE_TABLE_H_

struct slice_table  {
    uint8_t x;
    uint8_t y;
    uint16_t slice_size;
};

extern struct slice_table slice_tables[];

#endif
