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

#include "prores.h"
#include "debug.h"


void aprint_block(int16_t *block)
{

    int32_t x,y;
    for (y=0;y<8;y++) {
        for (x=0;x<8;x++) {
            printf("%d ", block[(y * 8) + x]);
        }
        printf("\n");
    }
    printf("\n");
}
void print_mb(int16_t *mb)
{
    int32_t i;
    for(i=0;i<4;i++) {
        aprint_block(mb + (i*64));
    }
        

}
void print_mb_cb(int16_t *mb)
{
    int32_t i;
    for(i=0;i<2;i++) {
        aprint_block(mb + (i*64));
    }
        

}
void print_slice_cb(int16_t *slice, int32_t mb_num)
{
    int32_t i;
    for(i=0;i<mb_num;i++) {
        print_mb_cb(slice + (i*64)*2);
    }
        

}
void print_slice(int16_t *slice, int32_t mb_num)
{
    int32_t i;
    for(i=0;i<mb_num;i++) {
        print_mb(slice + (i*64)*4);
    }
        

}
void print_pixels(int16_t *slice, int32_t mb_num)
{
    int32_t i;
    for(i=0;i<mb_num*4*64;i++) {
        printf("%d\n", slice[i]);
    }
        

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
