/**
 *
 * Copyright (c) 2020 Yuusuke Miyazaki
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 *
 **/
#ifndef _QSCALE_H_
#define _QSCALE_H_
#include <stdint.h>

struct qscale {
    uint8_t x;
    uint8_t y;
    uint8_t qscale;
};
extern struct qscale qscale_[];
#endif
