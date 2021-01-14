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
#include <math.h>

#include "config.h"
#include "prores.h"
#include "dct.h"




//static double kc_value[MAX_X][MAX_Y][MAX_X][MAX_Y];
#ifdef CUDA_ENCODER
__device__
#endif
int dct_block(int16_t *block, double *kc_value) {
    int h,v,i,x,y;
	double value;

    double result[MAX_X * MAX_Y];
    for (v=0;v<MAX_Y;v++) {
        for (h=0;h<MAX_X;h++) {
			value = 0;
		    for(y=0;y<MAX_Y;y++) {

        		for(x=0;x<MAX_X;x++) {
            		value += block[(y * 8) + x] *  kc_value[GET_KC_INDEX(x,y,h,v)];
        		}
    		}
    		if ( h == 0) {
#ifdef DEL_SQRT //changed quality
#ifdef DEL_DIVISION
        		value *= 0.70710678118;
#else
				//better quality
        		value *= 1/ 1.41421356237;
#endif

#else
        		value *= 1/ sqrt(2.0);
#endif
    		} else {
//        		value *= 1;
    		}
    		if (v == 0) {
#ifdef DEL_SQRT
#ifdef DEL_DIVISION
        		value *= 0.70710678118;
#else
				//better quality
        		value *= 1 / 1.41421356237;
#endif
#else
        		value *= 1/ sqrt(2.0);
#endif
    		} else {
//        		value *= 1;
    		}
			//double can't shift
    		value = value / 4;

            result[(v << 3) + h] = value;
        }
    }
    for(i = 0;i<MAX_X*MAX_Y;i++) {
        block[i] = (int16_t)result[i];
    }
    return 0;
}

void dct_init(double *kc_value)
{
    int h,v,x,y;
    for (v=0;v<MAX_Y;v++) {
        for (h=0;h<MAX_X;h++) {
		    for(y=0;y<MAX_Y;y++) {
        		for(x=0;x<MAX_X;x++) {
            		kc_value[GET_KC_INDEX(x,y,h,v)] = cos((M_PI * v * ((2.0 * y) + 1.0)) / 16.0) * cos((M_PI * h * ((2.0 * x) + 1.0)) / 16.0);
				}
			}
        }
    }
    return;
}
#ifdef CUDA_ENCODER
__device__
#endif

static void encode_qt(int16_t *block, uint8_t *qmat, int32_t  block_num)
{

    int16_t *data;
    int32_t i,j;
    for (i = 0; i < block_num; i++) {
        data = block + (i * BLOCK_IN_PIXEL);
        for (j=0;j<BLOCK_IN_PIXEL;j++) {
            data[j] = data [j] / ( qmat[j]) ;
        }

    }
}
#ifdef CUDA_ENCODER
__device__
#endif

static void encode_qscale(int16_t *block, uint8_t scale, int32_t  block_num)
{

    int16_t *data;
    int32_t i,j;
    for (i = 0; i < block_num; i++) {
        data = block + (i*BLOCK_IN_PIXEL);
        for (j=0;j<BLOCK_IN_PIXEL;j++) {
            data[j] = data [j] / scale;
        }

    }
}
#ifdef CUDA_ENCODER
__device__
#endif

static void pre_quant(int16_t *block, int32_t  block_num)
{

    int16_t *data;
    int32_t i,j;
    for (i = 0; i < block_num; i++) {
        data = block + (i*BLOCK_IN_PIXEL);
        for (j=0;j<BLOCK_IN_PIXEL;j++) {
            data[j] = data [j] << 3;
        }

    }
}
#ifdef CUDA_ENCODER
__device__
#endif

static void pre_dct(int16_t *block, int32_t  block_num)
{

    int16_t *data;
    int32_t i,j;
    for (i = 0; i < block_num; i++) {
        data = block + (i*BLOCK_IN_PIXEL);
        for (j=0;j<BLOCK_IN_PIXEL;j++) {
            data[j] = (data[j] >> 1) - 256;
        }

    }
}
// macro block num * block num per macro  block * pixel num per block * pixel size
// (mb_size(8) * MB_IN_BLOCK(4) * BLOCK_IN_PIXEL(64)
#ifdef CUDA_ENCODER
__gloval__
#endif

void dct_and_quant(int16_t *pixel, uint8_t *matrix, int slice_size_in_mb, int mb_in_block, double *kc_value, uint8_t qscale) {
    pre_dct(pixel, slice_size_in_mb * mb_in_block);

    int32_t i;
    for (i = 0;i< slice_size_in_mb * mb_in_block;i++) {
        dct_block(&pixel[i* BLOCK_IN_PIXEL],kc_value);
    }
    pre_quant(pixel, slice_size_in_mb * mb_in_block);
    encode_qt(pixel, matrix, slice_size_in_mb * mb_in_block);
    encode_qscale(pixel,qscale , slice_size_in_mb * mb_in_block);

}
