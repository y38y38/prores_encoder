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

#include "dct.h"
#include "bitstream.h"
#include "vlc.h"
#include "slice.h"



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


static void pre_dct(int16_t *block, int32_t  block_num)
{

    int16_t *data;
    int32_t i,j;
    for (i = 0; i < block_num; i++) {
        data = block + (i*BLOCK_IN_PIXEL);
        for (j=0;j<BLOCK_IN_PIXEL;j++) {
            //data[j] = (data[j] >> 1) - 256;
            data[j] = (data[j]) - 512;
        }

    }
}
#if 0
static void after_dct(int16_t *block, int32_t  block_num)
{
#if 0
    int16_t *data;
    int32_t i,j;
    for (i = 0; i < block_num; i++) {
        data = block + (i*BLOCK_IN_PIXEL);
        for (j=0;j<BLOCK_IN_PIXEL;j++) {
            data[j] = (data[j])>>1;
        }

    }
	#endif
}
static void pre_quant(int16_t *block, int32_t  block_num)
{

    int16_t *data;
    int32_t i,j;
    for (i = 0; i < block_num; i++) {
        data = block + (i*BLOCK_IN_PIXEL);
        for (j=0;j<BLOCK_IN_PIXEL;j++) {
            //data[j] = data [j] << 3;
            data[j] = data [j] << 2;
        }

    }
}
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
#endif
static void pre_quant_qt_qscale(int16_t *block, uint8_t *qmat, uint8_t scale, int32_t  block_num)
{
    int16_t *data;
    int32_t i,j;
    for (i = 0; i < block_num; i++) {
        data = block + (i * BLOCK_IN_PIXEL);
        for (j=0;j<BLOCK_IN_PIXEL;j++) {
            data[j] = (data [j] << 2) / (( qmat[j]) * scale) ;
        }

    }

}

// macro block num * block num per macro  block * pixel num per block * pixel size
// (mb_size(8) * MB_IN_BLOCK(4) * BLOCK_IN_PIXEL(64)

static uint32_t encode_slice_component(struct Slice *param, int16_t* pixel, uint8_t *matrix, int mb_in_block)
{
    uint32_t start_offset= getBitSize(param->bitstream);

    pre_dct(pixel, param->slice_size_in_mb * mb_in_block);

    int32_t i;
    for (i = 0;i< param->slice_size_in_mb * mb_in_block;i++) {
        dct_block(&pixel[i* BLOCK_IN_PIXEL]);
    }
    //after_dct(pixel, param->slice_size_in_mb * mb_in_block);
	pre_quant_qt_qscale(pixel, matrix,param->qscale,param->slice_size_in_mb * mb_in_block);

    //pre_quant(pixel, param->slice_size_in_mb * mb_in_block);
    //encode_qt(pixel, param->chroma_matrix, param->slice_size_in_mb * mb_in_block);
    //encode_qscale(pixel,param->qscale , param->slice_size_in_mb * mb_in_block);

    entropy_encode_dc_coefficients(pixel, param->slice_size_in_mb * mb_in_block, param->bitstream);
    entropy_encode_ac_coefficients(pixel, param->slice_size_in_mb * mb_in_block, param->bitstream);
    //byte aliened
    uint32_t size  = getBitSize(param->bitstream);
    if (size & 7 )  {
        setBit(param->bitstream, 0x0, 8 - (size % 8));
    }
    uint32_t current_offset = getBitSize(param->bitstream);
    return ((current_offset - start_offset)/8);
}



#if 0
static uint8_t qScale2quantization_index(uint8_t qscale)
{
    return qscale;
}
#endif

uint16_t encode_slice(struct Slice *param)
{
	//initBitStream(param->bitstream);

    uint32_t start_offset= getBitSize(param->bitstream);
//	uint32_t size2;
//	printf("start_slice_offset %d %p\n", start_offset, getBitStream(param->bitstream, &size2));
    uint8_t slice_header_size = 6;

    setBit(param->bitstream, slice_header_size , 5);

    uint8_t reserve =0x0;
    setBit(param->bitstream, reserve, 3);

    setByte(param->bitstream, &param->qscale, 1);

    uint32_t code_size_of_y_data_offset = getBitSize(param->bitstream);
    code_size_of_y_data_offset = code_size_of_y_data_offset >> 3;
    uint16_t size = 0;
    uint16_t coded_size_of_y_data = SET_DATA16(size);
    setByte(param->bitstream, (uint8_t*)&coded_size_of_y_data , 2);

    uint32_t code_size_of_cb_data_offset = getBitSize(param->bitstream);
    code_size_of_cb_data_offset = code_size_of_cb_data_offset >> 3 ;
    size = 0;
    uint16_t coded_size_of_cb_data = SET_DATA16(size);
    setByte(param->bitstream, (uint8_t*)&coded_size_of_cb_data , 2);
//	printf("offset=0x%x\n", code_size_of_cb_data_offset);

	getYver2((uint16_t*)param->working_buffer, param->y_data, param->mb_x,param->mb_y,param->slice_size_in_mb, param->horizontal, param->vertical);
	size = (uint16_t)encode_slice_component(param, param->working_buffer, param->luma_matrix, MB_IN_BLOCK);
    uint16_t y_size  = SET_DATA16(size);

    uint16_t cb_size;
    if (param->format_444 == true) {

		getYver2((uint16_t*)param->working_buffer, param->cb_data, param->mb_x,param->mb_y,param->slice_size_in_mb, param->horizontal, param->vertical);
		size = (uint16_t)encode_slice_component(param, (int16_t*)param->working_buffer, param->chroma_matrix, MB_IN_BLOCK);
        cb_size = SET_DATA16(size);


		getYver2((uint16_t*)param->working_buffer, param->cr_data, param->mb_x,param->mb_y,param->slice_size_in_mb, param->horizontal, param->vertical);
		size = (uint16_t)encode_slice_component(param, (int16_t*)param->working_buffer, param->chroma_matrix, MB_IN_BLOCK);

    } else {
		getCver2((uint16_t*)param->working_buffer, param->cb_data, param->mb_x,param->mb_y,param->slice_size_in_mb, param->horizontal, param->vertical);
		size = (uint16_t)encode_slice_component(param, (int16_t*)param->working_buffer, param->chroma_matrix, MB_422C_IN_BLCCK);
        cb_size = SET_DATA16(size);

		getCver2((uint16_t*)param->working_buffer, param->cr_data, param->mb_x,param->mb_y,param->slice_size_in_mb, param->horizontal, param->vertical);
		size = (uint16_t)encode_slice_component(param, (int16_t*)param->working_buffer, param->chroma_matrix, MB_422C_IN_BLCCK);
    }

    setByteInOffset(param->bitstream, code_size_of_y_data_offset , (uint8_t *)&y_size, 2);
    setByteInOffset(param->bitstream, code_size_of_cb_data_offset , (uint8_t *)&cb_size, 2);
    uint32_t current_offset = getBitSize(param->bitstream);
	//printf("size=0x%x\n",  ((current_offset - start_offset)/8));
    return ((current_offset - start_offset)/8);
}

