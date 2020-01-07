/**
 *
 * Copyright (c) 2020 Yuusuke Miyazaki
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 *
 **/
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define MAX_X   (8)
#define MAX_Y   (8)

static double dct(int16_t *block, int h, int v)
{
    int x=0,y=0;
    double value = 0;
    for(y=0;y<MAX_Y;y++) {
        for(x=0;x<MAX_X;x++) {
            double kc = cos((M_PI * v * ((2.0 * y) + 1.0)) / 16.0) * cos((M_PI * h * ((2.0 * x) + 1.0)) / 16.0);
            value += block[(y * 8) + x] *  kc;
        }
    }
    if ( h == 0) {
        value *= 1/ sqrt(2.0);
    } else {
        value *= 1;
    }
    if (v == 0) {
        value *= 1 / sqrt(2.0);
    } else {
        value *= 1;
    }

    value = value / 4;
    return value;
}
void print_block(int16_t *block)
{

    int x,y;
    for (y=0;y<MAX_Y;y++) {
        for (x=0;x<MAX_X;x++) {
            printf("%d ", block[(y * MAX_X) + x]);
        }
        printf("\n");
    }
    printf("\n");
}


int dct_block(int16_t *block) {
    int h,v,i;
    double result[MAX_X * MAX_Y];
    for (v=0;v<MAX_Y;v++) {
        for (h=0;h<MAX_X;h++) {
            result[(v * MAX_X) + h] = dct(block, h, v);
        }
    }
    for(i = 0;i<MAX_X*MAX_Y;i++) {
        block[i] = (int16_t)result[i];
    }

    return 0;
}
