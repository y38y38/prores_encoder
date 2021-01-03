/**
 *
 * Copyright (c) 2020 Yuusuke Miyazaki
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 *
 **/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>


#include "prores.h"
#include "encoder.h"
#include "config.h"
#include "dct.h"
#include "bitstream_cuda.h"
#include "vlc.h"
#include "slice.h"

#include "dct.cu"
#include "bitstream_cuda.cu"
#include "vlc.cu"


__device__
int mbXFormSliceNo(struct Slice_cuda* slice_param, int slice_no)
{
	uint32_t mb_x_max = (slice_param->horizontal + 15) >>4;
	int horizontal_slice_num = mb_x_max /slice_param->slice_size_in_mb;

	int mb_x = (slice_no % horizontal_slice_num) * slice_param->slice_size_in_mb;
	return mb_x;
}
__device__
int mbYFormSliceNo(struct Slice_cuda* slice_param, int slice_no)
{
	uint32_t mb_x_max = (slice_param->horizontal + 15) >>4;
	int horizontal_slice_num = mb_x_max /slice_param->slice_size_in_mb;

	int mb_y = slice_no / horizontal_slice_num;
	return mb_y;
}


__device__
static void getPixelblock(uint16_t *out, uint16_t *in, uint32_t x, uint32_t y, int32_t horizontal, int32_t vertical)
{
	//printf("%d %d %d %d\n", x,y, horizontal, vertical);
	int i;
	for(i=0;i<8;i++) {
		memcpy(out + (i*8),
		in + x + (horizontal * y) + (horizontal * i),
		8 * sizeof(uint16_t));
	//	printf(" %x\n", *(uint16_t*)(in + x + (horizontal * y) + (horizontal * i)));
	}
}

__device__
//get 1 slice data
static void getYver2(uint16_t *out, uint16_t *in, uint32_t mb_x, uint32_t mb_y, int32_t mb_size, int32_t horizontal, int32_t vertical)
{
	int i;
    int32_t block;
	int offset_x,offset_y;
    for (i=0;i<mb_size;i++) {
        for (block = 0 ; block < MB_IN_BLOCK;block++) {
			if (block == 0) {
				offset_x = 0;
				offset_y = 0;
			} else if (block == 1) {
				offset_x = 8;
				offset_y = 0;
			} else if (block == 2) {
				offset_x = 0;
				offset_y = 8;
			} else {
				offset_x = 8;
				offset_y = 8;
			}
			getPixelblock(out  +  i * 64 * 4 + (block * 64), in, (mb_x * 16) + (i * 16) + offset_x, (mb_y * 16) + offset_y, horizontal, vertical);
        }

    }
	return;
}

__device__
//get 1 slice data
static void getCver2(uint16_t *out, uint16_t *in, uint32_t mb_x, uint32_t mb_y, int32_t mb_size, int32_t horizontal, int32_t vertical)
{
	int i;
    int32_t block;
	int offset_x,offset_y;
    for (i=0;i<mb_size;i++) {
        for (block = 0 ; block < MB_422C_IN_BLCCK;block++) {
			if (block == 0) {
				offset_x = 0;
				offset_y = 0;
			} else {
				offset_x = 0;
				offset_y = 8;
			}
			getPixelblock(out  +  (i * 64 * 2) + (block * 64), in, (mb_x * 8) + (i * 8) + offset_x, (mb_y * 16) + offset_y, horizontal>>1, vertical);
        }

    }
	return;
}


__device__
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

__device__
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

__device__
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

__device__
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

__device__
static uint32_t encode_slice_component(struct Slice_cuda *param, int16_t* pixel, uint8_t *matrix, int mb_in_block, struct bitstream *bitstream, uint8_t qscale, double *kc_value)
{
    uint32_t start_offset= getBitSize_cuda(bitstream);

    pre_dct(pixel, param->slice_size_in_mb * mb_in_block);

    int32_t i;
    for (i = 0;i< param->slice_size_in_mb * mb_in_block;i++) {
        dct_block(&pixel[i* BLOCK_IN_PIXEL],kc_value);
    }
    pre_quant(pixel, param->slice_size_in_mb * mb_in_block);
    encode_qt(pixel, param->chroma_matrix, param->slice_size_in_mb * mb_in_block);
    encode_qscale(pixel,qscale , param->slice_size_in_mb * mb_in_block);

    entropy_encode_dc_coefficients(pixel, param->slice_size_in_mb * mb_in_block, bitstream);
    entropy_encode_ac_coefficients(pixel, param->slice_size_in_mb * mb_in_block, bitstream);
    //byte aliened
    uint32_t size  = getBitSize_cuda(bitstream);
    if (size & 7 )  {
        setBit_cuda(bitstream, 0x0, 8 - (size % 8));
    }
    uint32_t current_offset = getBitSize_cuda(bitstream);
    return ((current_offset - start_offset)/8);
}



#if 0
static uint8_t qScale2quantization_index(uint8_t qscale)
{
    return qscale;
}
#endif



__device__
int mbXFormSliceNo(struct Slice_cuda* slice_param, int slice_no);
__device__
int mbYFormSliceNo(struct Slice_cuda* slice_param,int slice_no);

__global__
void encode_slice(int slice_no, struct Slice_cuda * slice_param, uint8_t *qscale_table, uint16_t *y_data, uint16_t * cb_data, uint16_t * cr_data, struct bitstream *bitstream, uint16_t* slice_size_table, int16_t *buffer,  double * kc_value)

//void encode_slice(int slice_no, struct Slice_cuda * slice_param, uint8_t *qscale_table, uint16_t *y_data, uint16_t * cb_data, uint16_t * cr_data, struct bistream *bitstream, uint16_t* slice_size_table, int16_t *buffer)
{
	uint8_t *ptr = (uint8_t*)bitstream;
	struct bitstream *bitstream_ptr = (struct bitstream *)(ptr + ((sizeof(struct bitstream) + MAX_SLICE_BITSTREAM_SIZE) * slice_no));
//	struct bitstream *bitstream_ptr = &bitstream_ptr[slice_no];
	int16_t *working_buffer = (buffer + (slice_no * MAX_SLICE_DATA ));
	//printf("%p\n",bitstream_ptr);
	initBitStream_cuda(bitstream_ptr);

    uint32_t start_offset= getBitSize_cuda(bitstream_ptr);
//	uint32_t size2;
//	printf("start_slice_offset %d %p\n", start_offset, getBitStream(param->bitstream, &size2));
    uint8_t slice_header_size = 6;

    setBit_cuda(bitstream_ptr, slice_header_size , 5);
	//printf("%d %d\n", __LINE__, getBitSize(bitstream_ptr)/8);
    uint8_t reserve =0x0;
    setBit_cuda(bitstream_ptr, reserve, 3);

    setByte_cuda(bitstream_ptr, &qscale_table[slice_no], 1);
	//printf("%d %d\n", __LINE__, getBitSize(bitstream_ptr)/8);

    uint32_t code_size_of_y_data_offset = getBitSize_cuda(bitstream_ptr);
    code_size_of_y_data_offset = code_size_of_y_data_offset >> 3;
    uint16_t size = 0;
    uint16_t coded_size_of_y_data = SET_DATA16(size);
    setByte_cuda(bitstream_ptr, (uint8_t*)&coded_size_of_y_data , 2);
	//printf("%d %d\n", __LINE__, getBitSize(bitstream_ptr)/8);

    uint32_t code_size_of_cb_data_offset = getBitSize_cuda(bitstream_ptr);
    code_size_of_cb_data_offset = code_size_of_cb_data_offset >> 3 ;
    size = 0;
    uint16_t coded_size_of_cb_data = SET_DATA16(size);
    setByte_cuda(bitstream_ptr, (uint8_t*)&coded_size_of_cb_data , 2);
	//printf("%d %d\n", __LINE__, getBitSize(bitstream_ptr)/8);

	int mb_x = mbXFormSliceNo(slice_param, slice_no);
	int mb_y = mbYFormSliceNo(slice_param, slice_no);
	//printf(" x %x y %x ", mb_x, mb_y);
	getYver2((uint16_t*)working_buffer, y_data, mb_x, mb_y,slice_param->slice_size_in_mb, slice_param->horizontal, slice_param->vertical);
	size = (uint16_t)encode_slice_component(slice_param, working_buffer, slice_param->luma_matrix, MB_IN_BLOCK,bitstream_ptr, qscale_table[slice_no], kc_value);
    uint16_t y_size  = SET_DATA16(size);
	//printf("ysize=0x%x\n", y_size);
    uint16_t cb_size;
    if (slice_param->format_444 == true) {

		getYver2((uint16_t*)working_buffer, cb_data, mb_x,mb_y,slice_param->slice_size_in_mb, slice_param->horizontal, slice_param->vertical);
		size = (uint16_t)encode_slice_component(slice_param, (int16_t*)working_buffer, slice_param->chroma_matrix, MB_IN_BLOCK, bitstream_ptr,qscale_table[slice_no], kc_value);
        cb_size = SET_DATA16(size);


		getYver2((uint16_t*)working_buffer, cr_data, mb_x,mb_y,slice_param->slice_size_in_mb, slice_param->horizontal, slice_param->vertical);
		size = (uint16_t)encode_slice_component(slice_param, (int16_t*)working_buffer, slice_param->chroma_matrix, MB_IN_BLOCK, bitstream_ptr,qscale_table[slice_no], kc_value);

    } else {
		
		getCver2((uint16_t*)working_buffer, cb_data, mb_x,mb_y,slice_param->slice_size_in_mb, slice_param->horizontal, slice_param->vertical);
#if 0
		for(int i=0;i<128;i++) {
			printf("%x ", cb_data[i]);
		}
		printf("%d %d \n\n", slice_param->slice_size_in_mb, slice_param->horizontal, slice_param->vertical);
		for(int i=0;i<128;i++) {
			printf("%x ", working_buffer[i]);
		}
#endif
		size = (uint16_t)encode_slice_component(slice_param, (int16_t*)working_buffer, slice_param->chroma_matrix, MB_422C_IN_BLCCK, bitstream_ptr,qscale_table[slice_no], kc_value);
        cb_size = SET_DATA16(size);
		//printf("cbsize=0x%x %d %d\n", cb_size, mb_x,mb_y);

		getCver2((uint16_t*)working_buffer, cr_data, mb_x,mb_y,slice_param->slice_size_in_mb, slice_param->horizontal, slice_param->vertical);
		size = (uint16_t)encode_slice_component(slice_param, (int16_t*)working_buffer, slice_param->chroma_matrix, MB_422C_IN_BLCCK, bitstream_ptr,qscale_table[slice_no], kc_value);
    }

    setByteInOffset_cuda(bitstream_ptr, code_size_of_y_data_offset , (uint8_t *)&y_size, 2);
    setByteInOffset_cuda(bitstream_ptr, code_size_of_cb_data_offset , (uint8_t *)&cb_size, 2);
    uint32_t current_offset = getBitSize_cuda(bitstream_ptr);
	slice_size_table[slice_no] = ((current_offset - start_offset)/8);
	printf("\n%x\n",bitstream_ptr->bitstream_buffer);
		for(int j=0;j<128;j++) {
			printf("%x ", bitstream_ptr->bitstream_buffer[j]);
		}

	//printf("size = 0x%x\n", ((current_offset - start_offset)/8));
    return;
}


