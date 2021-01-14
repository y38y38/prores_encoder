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
#include "bitstream.h"
#include "vlc.h"
#include "slice.h"



int mbXFormSliceNo(struct Slice_cuda* slice_param, int slice_no)
{
	uint32_t mb_x_max = (slice_param->horizontal + 15) >>4;
	int horizontal_slice_num = mb_x_max /slice_param->slice_size_in_mb;

	int mb_x = (slice_no % horizontal_slice_num) * slice_param->slice_size_in_mb;
	return mb_x;
}

int mbYFormSliceNo(struct Slice_cuda* slice_param, int slice_no)
{
	uint32_t mb_x_max = (slice_param->horizontal + 15) >>4;
	int horizontal_slice_num = mb_x_max /slice_param->slice_size_in_mb;

	int mb_y = slice_no / horizontal_slice_num;
	return mb_y;
}


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
void getYver2(uint16_t *out, uint16_t *in, uint32_t mb_x, uint32_t mb_y, int32_t mb_size, int32_t horizontal, int32_t vertical)
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
void getCver2(uint16_t *out, uint16_t *in, uint32_t mb_x, uint32_t mb_y, int32_t mb_size, int32_t horizontal, int32_t vertical)
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



static uint32_t encode_slice_component(struct Slice_cuda *param, int16_t* pixel, uint8_t *matrix, int mb_in_block, struct bitstream *bitstream, uint8_t qscale, double *kc_value)
{
    uint32_t start_offset= getBitSize(bitstream);

#if 0
    pre_dct(pixel, param->slice_size_in_mb * mb_in_block);

    int32_t i;
    for (i = 0;i< param->slice_size_in_mb * mb_in_block;i++) {
        dct_block(&pixel[i* BLOCK_IN_PIXEL],kc_value);
    }
    pre_quant(pixel, param->slice_size_in_mb * mb_in_block);
    encode_qt(pixel, param->chroma_matrix, param->slice_size_in_mb * mb_in_block);
    encode_qscale(pixel,qscale , param->slice_size_in_mb * mb_in_block);
#else
	//dct_and_quant(pixel, matrix, param->slice_size_in_mb, mb_in_block, kc_value, qscale);
#endif
    entropy_encode_dc_coefficients(pixel, param->slice_size_in_mb * mb_in_block, bitstream);
    entropy_encode_ac_coefficients(pixel, param->slice_size_in_mb * mb_in_block, bitstream);
    //byte aliened
    uint32_t size  = getBitSize(bitstream);
    if (size & 7 )  {
        setBit(bitstream, 0x0, 8 - (size % 8));
    }
    uint32_t current_offset = getBitSize(bitstream);
    return ((current_offset - start_offset)/8);
}



#if 0
static uint8_t qScale2quantization_index(uint8_t qscale)
{
    return qscale;
}
#endif


void encode_slices2(struct Slice_cuda *param, int slice_no, uint16_t * slice_size_table, struct bitstream *bitstream, int16_t*working_buffer, double *kc_value)

{
	struct bitstream *bitstream_ptr = bitstream;
	//uint8_t *ptr = (uint8_t*)bitstream;
	//struct bitstream *bitstream_ptr = (struct bitstream *)(ptr + ((sizeof(struct bitstream) + MAX_SLICE_BITSTREAM_SIZE) * slice_no));
//	struct bitstream *bitstream_ptr = &bitstream_ptr[slice_no];
	//int16_t *working_buffer = (buffer + (slice_no * MAX_SLICE_DATA ));
	//printf("%p\n",bitstream_ptr);
	//initBitStream(bitstream_ptr);

    uint32_t start_offset= getBitSize(bitstream_ptr);
//	uint32_t size2;
//	printf("start_slice_offset %d %p\n", start_offset, getBitStream(param->bitstream, &size2));
    uint8_t slice_header_size = 6;

    setBit(bitstream_ptr, slice_header_size , 5);
	//printf("%d %d\n", __LINE__, getBitSize(bitstream_ptr)/8);
    uint8_t reserve =0x0;
    setBit(bitstream_ptr, reserve, 3);

    setByte(bitstream_ptr, &param->qscale_table[slice_no], 1);
	//printf("%d %d\n", __LINE__, getBitSize(bitstream_ptr)/8);

    uint32_t code_size_of_y_data_offset = getBitSize(bitstream_ptr);
    code_size_of_y_data_offset = code_size_of_y_data_offset >> 3;
    uint16_t size = 0;
    uint16_t coded_size_of_y_data = SET_DATA16(size);
    setByte(bitstream_ptr, (uint8_t*)&coded_size_of_y_data , 2);
	//printf("%d %d\n", __LINE__, getBitSize(bitstream_ptr)/8);

    uint32_t code_size_of_cb_data_offset = getBitSize(bitstream_ptr);
    code_size_of_cb_data_offset = code_size_of_cb_data_offset >> 3 ;
    size = 0;
    uint16_t coded_size_of_cb_data = SET_DATA16(size);
    setByte(bitstream_ptr, (uint8_t*)&coded_size_of_cb_data , 2);
	//printf("%d %d\n", __LINE__, getBitSize(bitstream_ptr)/8);

	int cb_offset = MAX_SLICE_DATA  * param->slice_num_max;
	int cr_offset = (MAX_SLICE_DATA * param->slice_num_max) * 2 ;

	int mb_x = mbXFormSliceNo(param, slice_no);
	int mb_y = mbYFormSliceNo(param, slice_no);
	//printf(" x %x y %x ", mb_x, mb_y);
	//getYver2((uint16_t*)working_buffer, param->y_data, mb_x, mb_y,param->slice_size_in_mb, param->horizontal, param->vertical);
	size = (uint16_t)encode_slice_component(param, working_buffer + (slice_no *(MAX_SLICE_DATA )), param->luma_matrix, MB_IN_BLOCK,bitstream_ptr, param->qscale_table[slice_no], kc_value);
    uint16_t y_size  = SET_DATA16(size);
	//printf("ysize=0x%x\n", y_size);
    uint16_t cb_size;
    if (param->format_444 == true) {

		//getYver2((uint16_t*)working_buffer, param->cb_data, mb_x,mb_y,param->slice_size_in_mb, param->horizontal, param->vertical);
		size = (uint16_t)encode_slice_component(param, (int16_t*)(working_buffer + cb_offset + (slice_no *(MAX_SLICE_DATA ))), param->chroma_matrix, MB_IN_BLOCK, bitstream_ptr,param->qscale_table[slice_no], kc_value);
        cb_size = SET_DATA16(size);


		//getYver2((uint16_t*)working_buffer, param->cr_data, mb_x,mb_y,param->slice_size_in_mb, param->horizontal, param->vertical);
		size = (uint16_t)encode_slice_component(param, (int16_t*)(working_buffer+ cr_offset + (slice_no*(MAX_SLICE_DATA ))), param->chroma_matrix, MB_IN_BLOCK, bitstream_ptr,param->qscale_table[slice_no], kc_value);

    } else {
		
		//getCver2((uint16_t*)working_buffer, param->cb_data, mb_x,mb_y,param->slice_size_in_mb, param->horizontal, param->vertical);
#if 0
		for(int i=0;i<128;i++) {
			printf("%x ", cb_data[i]);
		}
		printf("%d %d \n\n", param->slice_size_in_mb, param->horizontal, param->vertical);
		for(int i=0;i<128;i++) {
			printf("%x ", working_buffer[i]);
		}
#endif
		size = (uint16_t)encode_slice_component(param, (int16_t*)(working_buffer+ cb_offset + (slice_no*(MAX_SLICE_DATA ))), param->chroma_matrix, MB_422C_IN_BLCCK, bitstream_ptr,param->qscale_table[slice_no], kc_value);
        cb_size = SET_DATA16(size);
		//printf("cbsize=0x%x %d %d\n", cb_size, mb_x,mb_y);

		//getCver2((uint16_t*)working_buffer, param->cr_data, mb_x,mb_y,param->slice_size_in_mb, param->horizontal, param->vertical);
		size = (uint16_t)encode_slice_component(param, (int16_t*)(working_buffer+ cr_offset + (slice_no*(MAX_SLICE_DATA ))), param->chroma_matrix, MB_422C_IN_BLCCK, bitstream_ptr,param->qscale_table[slice_no], kc_value);
    }

    setByteInOffset(bitstream_ptr, code_size_of_y_data_offset , (uint8_t *)&y_size, 2);
    setByteInOffset(bitstream_ptr, code_size_of_cb_data_offset , (uint8_t *)&cb_size, 2);
    uint32_t current_offset = getBitSize(bitstream_ptr);
	slice_size_table[slice_no] = ((current_offset - start_offset)/8);
	//printf("\n%x\n",bitstream_ptr->bitstream_buffer);
#if 0
		for(int j=0;j<128;j++) {
			printf("%x ", bitstream_ptr->bitstream_buffer[j]);
		}
#endif
	//printf("size = 0x%x\n", ((current_offset - start_offset)/8));
    return;
}


