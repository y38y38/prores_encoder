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
__global__
void dct_and_quant(int16_t *working_buffer, uint8_t *matrix, int slice_size_in_mb, double *kc_value, uint8_t *qscale_table, int slice_num_max) {
#else
//void dct_and_quant(int ix, int16_t *working_buffer, uint8_t *matrix, int slice_size_in_mb, uint32_t *mb_size, double *kc_value, uint8_t* qscale_table, int slice_num_max) {
void dct_and_quant(int ix, int16_t *working_buffer, uint8_t *matrix, int slice_size_in_mb, uint32_t *mb_size, double *kc_value, uint8_t *qscale_table, int slice_num_max) {
#endif

#ifdef CUDA_ENCODER
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
#endif
#if 0
	int ii = ix % slice_num_max;
	int j =  ix / slice_num_max;
	int cb_offset = MAX_SLICE_DATA * slice_num_max;
//	int16_t *pixel = working_buffer + (j*cb_offset) + (ii *(MAX_SLICE_DATA));
	int16_t *pixel = working_buffer;
	int mb_in_block  = mb_size[j];


	printf("%d %d %d %d\n", ix, mb_in_block,j, ii);
 
#else
	int ii = ix % slice_num_max;
	int j =  ix / slice_num_max;
//	int mb_in_block  = mb_size;
	int mb_in_block  = mb_size[j];
	uint8_t qscale = qscale_table[ii];
//	int16_t *pixel = working_buffer;
	int cb_offset = MAX_SLICE_DATA * slice_num_max;
	int16_t *pixel = working_buffer + (j*cb_offset) + (ii *(MAX_SLICE_DATA));
#endif

    pre_dct(pixel, slice_size_in_mb * mb_in_block);

    int32_t i;
    for (i = 0;i< slice_size_in_mb * mb_in_block;i++) {
        dct_block(&pixel[i* BLOCK_IN_PIXEL],kc_value);
    }
    pre_quant(pixel, slice_size_in_mb * mb_in_block);
	//printf("ix=%d j=%d ii=%d slice_num_max=%d %p %p\n",ix, j, ii, slice_num_max, matrix, matrix +(j*64));
    encode_qt(pixel, matrix + (j * 64), slice_size_in_mb * mb_in_block);
    encode_qscale(pixel,qscale , slice_size_in_mb * mb_in_block);
}
