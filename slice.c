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
//#include <pthread.h>

#include "dct.h"
#include "bitstream.h"
#include "vlc.h"
#include "slice.h"
#include "encoder.h"

#define MB_IN_BLOCK                   (4)
#define MB_422C_IN_BLCCK              (2)
#define BLOCK_IN_PIXEL               (64)
#define MB_HORIZONTAL_Y_IN_PIXEL     (16)
#define MB_HORIZONTAL_422C_IN_PIXEL   (8)
#define MB_VERTIVAL_IN_PIXEL         (16)
#define MAX_MB_SIZE_IN_MB             (8)
#define BLOCK_HORIZONTAL_IN_PIXEL     (8)
#define BLOCK_VERTIVAL_IN_PIXEL       (8)


void write_slice_size(int slice_no, int size);


//pthread_mutex_t end_frame_mutex;
//void start_write_next_bitstream(struct thread_param * param);
//void wait_write_bitstream(struct thread_param * param);


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



static void getYver2block(uint16_t *out, uint16_t *in, uint32_t x, uint32_t y, int32_t horizontal, int32_t vertical)
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
			getYver2block(out  +  i * 64 * 4 + (block * 64), in, (mb_x * 16) + (i * 16) + offset_x, (mb_y * 16) + offset_y, horizontal, vertical);
        }

    }
	return;
}

static void getCver2block(uint16_t *out, uint16_t *in, uint32_t x, uint32_t y, int32_t horizontal, int32_t vertical)
{
	//printf("%d %d %d %d\n", x,y, horizontal, vertical);
	int i;
	for(i=0;i<8;i++) {
		memcpy(out + (i*8),
		in + x + (horizontal * y) + (horizontal * i),
		8 * sizeof(uint16_t));
		//printf(" %x\n", *(uint16_t*)(in + x + (horizontal * y) + (horizontal * i)));
	}
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
			getCver2block(out  +  (i * 64 * 2) + (block * 64), in, (mb_x * 8) + (i * 8) + offset_x, (mb_y * 16) + offset_y, horizontal>>1, vertical);
        }

    }
	return;
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

static uint32_t encode_slice_y(uint16_t*y_data, uint32_t mb_x, uint32_t mb_y, int32_t scale, uint8_t *matrix, uint32_t slice_size_in_mb, int horizontal, int vertical, struct bitstream * bitstream, struct thread_param *param)
{
    uint32_t start_offset= getBitSize(bitstream);

	getYver2((uint16_t*)param->y_slice, y_data, mb_x,mb_y,slice_size_in_mb, horizontal, vertical);

    pre_dct(param->y_slice, slice_size_in_mb * MB_IN_BLOCK);

    int32_t i;
    for (i = 0;i< slice_size_in_mb * MB_IN_BLOCK;i++) {
        dct_block(&param->y_slice[i * BLOCK_IN_PIXEL]);
    }

    pre_quant(param->y_slice, slice_size_in_mb * MB_IN_BLOCK);

    encode_qt(param->y_slice, matrix, slice_size_in_mb * MB_IN_BLOCK);
    encode_qscale(param->y_slice, scale , slice_size_in_mb * MB_IN_BLOCK);

    entropy_encode_dc_coefficients(param->y_slice, slice_size_in_mb * MB_IN_BLOCK, bitstream);
    entropy_encode_ac_coefficients(param->y_slice, slice_size_in_mb * MB_IN_BLOCK, bitstream);

    //byte aliened
    uint32_t size  = getBitSize(bitstream);
    if (size & 7 )  {
        setBit(bitstream, 0x0, 8 - (size & 7));
    }
    uint32_t current_offset = getBitSize(bitstream);

    return ((current_offset - start_offset)/8);
}
static uint32_t encode_slice_cb(uint16_t*cb_data, uint32_t mb_x, uint32_t mb_y, int32_t scale, uint8_t *matrix, uint32_t slice_size_in_mb, int horizontal, int vertical, struct bitstream *bitstream, struct thread_param *param)
{
    //printf("cb start\n");
    uint32_t start_offset= getBitSize(bitstream);

	getCver2((uint16_t*)param->cb_slice, cb_data, mb_x,mb_y,slice_size_in_mb, horizontal, vertical);

    pre_dct(param->cb_slice, slice_size_in_mb * MB_422C_IN_BLCCK);

    int32_t i;
    for (i = 0;i< slice_size_in_mb * MB_422C_IN_BLCCK;i++) {
        dct_block(&param->cb_slice[i* BLOCK_IN_PIXEL]);
    }

    pre_quant(param->cb_slice, slice_size_in_mb * MB_422C_IN_BLCCK);

    encode_qt(param->cb_slice, matrix, slice_size_in_mb * MB_422C_IN_BLCCK);
    encode_qscale(param->cb_slice,scale , slice_size_in_mb * MB_422C_IN_BLCCK);

    entropy_encode_dc_coefficients(param->cb_slice, slice_size_in_mb * MB_422C_IN_BLCCK, bitstream);
    entropy_encode_ac_coefficients(param->cb_slice, slice_size_in_mb * MB_422C_IN_BLCCK, bitstream);

    //byte aliened
    uint32_t size  = getBitSize(bitstream);
    if (size & 0x7 )  {
        setBit(bitstream, 0x0, 8 - (size % 8));
    }
    uint32_t current_offset = getBitSize(bitstream);
    return ((current_offset - start_offset)/8);
}
static uint32_t encode_slice_cr(uint16_t*cr_data, uint32_t mb_x, uint32_t mb_y, int32_t scale, uint8_t *matrix, uint32_t slice_size_in_mb, int horizontal, int vertical, struct bitstream *bitstream, struct thread_param *param)
{
    //printf("%s start\n", __FUNCTION__);
    uint32_t start_offset= getBitSize(bitstream);

	getCver2((uint16_t*)param->cr_slice, cr_data, mb_x,mb_y,slice_size_in_mb, horizontal, vertical);

    pre_dct(param->cr_slice, slice_size_in_mb * MB_422C_IN_BLCCK);

    int32_t i;
    for (i = 0;i< slice_size_in_mb * MB_422C_IN_BLCCK;i++) {
        dct_block(&param->cr_slice[i* BLOCK_IN_PIXEL]);
    }
    //print_slice_cb(cr_slice, 4);
    pre_quant(param->cr_slice, slice_size_in_mb * MB_422C_IN_BLCCK);
    encode_qt(param->cr_slice, matrix, slice_size_in_mb * MB_422C_IN_BLCCK);
    encode_qscale(param->cr_slice,scale , slice_size_in_mb * MB_422C_IN_BLCCK);

    entropy_encode_dc_coefficients(param->cr_slice, slice_size_in_mb * MB_422C_IN_BLCCK, bitstream);
    entropy_encode_ac_coefficients(param->cr_slice, slice_size_in_mb * MB_422C_IN_BLCCK, bitstream);
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

void encode_slice(struct Slice *param)
{
	initBitStream(param->bitstream);

    uint32_t start_offset= getBitSize(param->bitstream);

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

    size = (uint16_t)encode_slice_y(param->y_data, param->mb_x, param->mb_y, param->qscale, param->luma_matrix, param->slice_size_in_mb, param->horizontal, param->vertical, param->bitstream, param->thread_param);

    uint16_t y_size  = SET_DATA16(size);
    uint16_t cb_size;
    if (param->format_444 == true) {
        size = (uint16_t)encode_slice_y(param->cb_data, param->mb_x, param->mb_y, param->qscale, param->chroma_matrix, param->slice_size_in_mb, param->horizontal, param->vertical, param->bitstream, param->thread_param);
        cb_size = SET_DATA16(size);
        size = (uint16_t)encode_slice_y(param->cr_data, param->mb_x, param->mb_y, param->qscale, param->chroma_matrix, param->slice_size_in_mb, param->horizontal, param->vertical, param->bitstream, param->thread_param);
    } else {
        size = (uint16_t)encode_slice_cb(param->cb_data, param->mb_x, param->mb_y, param->qscale, param->chroma_matrix, param->slice_size_in_mb, param->horizontal, param->vertical, param->bitstream, param->thread_param);
        cb_size = SET_DATA16(size);
        size = (uint16_t)encode_slice_cr(param->cr_data, param->mb_x, param->mb_y, param->qscale, param->chroma_matrix, param->slice_size_in_mb, param->horizontal, param->vertical, param->bitstream, param->thread_param);
    }

    setByteInOffset(param->bitstream, code_size_of_y_data_offset , (uint8_t *)&y_size, 2);
    setByteInOffset(param->bitstream, code_size_of_cb_data_offset , (uint8_t *)&cb_size, 2);
    uint32_t current_offset = getBitSize(param->bitstream);

	write_slice_size(param->slice_no, ((current_offset - start_offset)/8));
    return;
}

