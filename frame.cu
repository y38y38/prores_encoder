/**
 *
 * Copyright (c) 2020 Yuusuke Miyazaki
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 *
 **/

#include "cuda_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
//#include <math.h>
#include <stdbool.h>
#include <pthread.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <time.h>
#include <sys/time.h>


#include "config.h"
#include "dct.h"
#include "dct_init.h"
#include "bitstream.h"
#include "vlc.h"
#include "slice.h"
#include "encoder.h"
#include "debug.h"



#define MACRO_BLOCK_Y_HORIZONTAL  (16)
#define MACRO_BLOCK_Y_VERTICAL    (16)

#define MACRO_BLOCK_422_C_HORIZONTAL  (8)
#define MACRO_BLOCK_422_C_VERTICAL    (16)

#define MAX_BITSTREAM_SIZE	(1073741824) //1M



static struct bitstream *write_bitstream;


static uint16_t slice_size_table[MAX_SLICE_NUM];

uint32_t slice_num_max;

static uint32_t picture_size_offset_ = 0;


int32_t GetSliceNum(int32_t horizontal, int32_t vertical, int32_t sliceSize)
{
    int32_t mb_x_max = (horizontal + 15)  >> 4;
    int32_t mb_y_max = (vertical + 15) >> 4;


    int32_t slice_num_max_tmp;

    int32_t numMbsRemainingInRow = mb_x_max;
    int32_t number_of_slices_per_mb_row;
    int j = 0;

    do {
        while (numMbsRemainingInRow >= sliceSize) {
            j++;
            numMbsRemainingInRow  -=sliceSize;

        }
        sliceSize >>= 1;
    } while(numMbsRemainingInRow  > 0);

    number_of_slices_per_mb_row = j;

    slice_num_max_tmp = number_of_slices_per_mb_row * mb_y_max;
    return slice_num_max_tmp;

}
uint32_t GetEncodeHorizontal(int32_t horizontal)
{
    return ((horizontal + 15)  >> 4) << 4;

}
uint32_t GetEncodeVertical(int32_t vertical)
{
    return ((vertical + 15)  >> 4) << 4;
}



void set_picture_header(struct encoder_param* param)
{

    uint8_t picture_header_size = 0x8;
    setBit(write_bitstream, picture_header_size, 5);

    uint8_t reserved = 0x0;
    setBit(write_bitstream, reserved , 3);

    picture_size_offset_ = (getBitSize(write_bitstream)) >> 3 ;

    uint32_t picture_size = SET_DATA32(0);
    setByte(write_bitstream, (uint8_t*)&picture_size, 4);

    uint32_t slice_num = GetSliceNum(param->horizontal, param->vertical, param->slice_size_in_mb);
    uint16_t deprecated_number_of_slices =  SET_DATA16(slice_num);
    setByte(write_bitstream, (uint8_t*)&deprecated_number_of_slices , 0x2);


    uint8_t reserved2 = 0x0;
    setBit(write_bitstream, reserved2 , 2);

    uint8_t log2_desired_slice_size_in_mb;
    if (param->slice_size_in_mb == 1) {
        log2_desired_slice_size_in_mb = 0;
    } else if (param->slice_size_in_mb == 2) {
        log2_desired_slice_size_in_mb = 1;
    } else if (param->slice_size_in_mb == 4) {
        log2_desired_slice_size_in_mb = 2;
    } else {
        log2_desired_slice_size_in_mb = 3;
    }
    setBit(write_bitstream,log2_desired_slice_size_in_mb, 2);

    uint8_t reserved3 = 0x0;
    setBit(write_bitstream, reserved3 , 4);


}
void set_frame_header(struct encoder_param* param)
{
    uint16_t frame_header_size = SET_DATA16(0x94);
    setByte(write_bitstream, (uint8_t*)&frame_header_size, 0x2);

    uint8_t reserved = 0x0;
    setByte(write_bitstream, &reserved, 0x1);

    uint8_t bitstream_version = 0x0;
    setByte(write_bitstream, &bitstream_version, 0x1);


    uint32_t encoder_identifier = SET_DATA32(0x4c617663);
    setByte(write_bitstream, (uint8_t*)&encoder_identifier, 0x4);

    uint16_t horizontal_size = SET_DATA16(param->horizontal);
    setByte(write_bitstream, (uint8_t*)&horizontal_size , 0x2);

    uint16_t vertical_size = SET_DATA16(param->vertical);
    setByte(write_bitstream, (uint8_t*)&vertical_size, 0x2);


    uint8_t chroma_format;
    if (param->format_444 == true) {
        chroma_format = 0x3;
    } else {
        chroma_format = 0x2;
    }
    setBit(write_bitstream, chroma_format, 2);

    uint8_t reserved1 = 0x0;
    setBit(write_bitstream, reserved1, 2);

    uint8_t interlace_mode = 0;
    setBit(write_bitstream, interlace_mode, 2);

    uint8_t reserved2 = 0x0;
    setBit(write_bitstream, reserved2, 2);

    uint8_t aspect_ratio_information = 0;
    setBit(write_bitstream, aspect_ratio_information, 4);

    uint8_t frame_rate_code = 0;
    setBit(write_bitstream, frame_rate_code, 4);

    uint8_t color_primaries = 0x0;
    setByte(write_bitstream, &color_primaries, 1);

    uint8_t transfer_characteristic = 0x0;
    setByte(write_bitstream, &transfer_characteristic , 1);

    uint8_t matrix_coefficients = 0x2;
    setByte(write_bitstream, &matrix_coefficients, 1);


    uint8_t reserved3 = 0x4;
    setBit(write_bitstream, reserved3 , 4);

    //printf("1   %x %x\n", tmp_buf_byte_offset, tmp_buf[0x1b]);
    uint8_t alpha_channel_type = 0x0;
    setBit(write_bitstream, alpha_channel_type , 4);

    //printf("2   %x %x\n", tmp_buf_byte_offset, tmp_buf[0x1b]);
    uint8_t reserved4 = 0x0;
    setByte(write_bitstream, &reserved4 , 1);
    
    //printf("3   %x %x\n", tmp_buf_byte_offset, tmp_buf[0x1b]);
    uint8_t reserved5 = 0x0;
    setBit(write_bitstream, reserved5, 6);
    
    //printf("4   %x %x\n", tmp_buf_byte_offset, tmp_buf[0x1b]);
    uint8_t load_luma_quantization_matrix = 0x1;
    setBit(write_bitstream, load_luma_quantization_matrix, 1);

    //printf("5   %x %x\n", tmp_buf_byte_offset, tmp_buf[0x1b]);

    uint8_t load_chroma_quantization_matrix = 0x1;
    setBit(write_bitstream, load_chroma_quantization_matrix, 1);

    setByte(write_bitstream, param->luma_matrix, MATRIX_NUM );
    setByte(write_bitstream, param->chroma_matrix, MATRIX_NUM );


}

void setSliceTalbeFlush(uint16_t size, uint32_t offset) {
    uint16_t slice_size = SET_DATA16(size);
    setByteInOffset(write_bitstream, offset, (uint8_t*)&slice_size, 2);
    

}



struct Slice_cuda h_slice_param_cuda;
//uint8_t h_qscale_table_cuda[MAX_SLICE_NUM];

double h_kc_value[KC_INDEX_MAX];


cudaError_t cudaMallocAndCpy(void ** dst, void *src, int size)
{
	cudaError_t err;
	err = cudaMalloc(dst, size);
	if (err != cudaSuccess) {
		printf("%d\n", __LINE__);
		return err;
	}
	err = cudaMemcpy(*dst, src, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%d\n", __LINE__);
		return err;
	}
	return err;
}

void encode_slices(struct encoder_param * param)
{
    slice_num_max = GetSliceNum(param->horizontal, param->vertical, param->slice_size_in_mb);

    /* write dummy slice size table */
    int32_t i;
    uint32_t slice_size_table_offset = (getBitSize(write_bitstream)) >> 3 ;
    for (i = 0; i < slice_num_max ; i++) {
        uint16_t slice_size = 0x0;
        setByte(write_bitstream, (uint8_t*)&slice_size, 2);
    }
	h_slice_param_cuda.luma_matrix = param->luma_matrix;
	h_slice_param_cuda.chroma_matrix = param->chroma_matrix;
    h_slice_param_cuda.slice_size_in_mb= param->slice_size_in_mb;
    h_slice_param_cuda.horizontal= param->horizontal;
    h_slice_param_cuda.vertical= param->vertical;
    h_slice_param_cuda.format_444 = param->format_444;
	h_slice_param_cuda.slice_num_max = slice_num_max;
	h_slice_param_cuda.qscale_table = param->qscale_table;
	h_slice_param_cuda.y_data = param->y_data;
	h_slice_param_cuda.cb_data = param->cb_data;
	h_slice_param_cuda.cr_data = param->cr_data;

#if 1
	int16_t *working_buffer;
	int working_buffer_size = (MAX_SLICE_DATA * sizeof(uint16_t)) * 3 * slice_num_max;
	working_buffer = (int16_t*)malloc(working_buffer_size);
	if (working_buffer == NULL ) {
		printf("malloc error %d", __LINE__);
	}
#endif

#if 0
	double *c_kc_value;
	int kc_value_size = sizeof(double) * KC_INDEX_MAX;
	c_kc_value = (double*)malloc(kc_value_size);
	if (c_kc_value == NULL ) {
		printf("cudaMalloc error %d", __LINE__);
	}
	memcpy(c_kc_value, h_kc_value, kc_value_size);
#endif

#if 1
	int cb_offset = MAX_SLICE_DATA  * slice_num_max;
	int cr_offset = (MAX_SLICE_DATA * slice_num_max) * 2 ;
	for (i = 0; i<slice_num_max;i++) {
		int mb_x = mbXFormSliceNo(&h_slice_param_cuda, i);
		int mb_y = mbYFormSliceNo(&h_slice_param_cuda, i);
		getYver2((uint16_t*)(working_buffer + (i*(MAX_SLICE_DATA ))), h_slice_param_cuda.y_data, mb_x, mb_y,h_slice_param_cuda.slice_size_in_mb, h_slice_param_cuda.horizontal, h_slice_param_cuda.vertical);
	}
	if (h_slice_param_cuda.format_444 == true) {
		for (i = 0; i<slice_num_max;i++) {
			int mb_x = mbXFormSliceNo(&h_slice_param_cuda, i);
			int mb_y = mbYFormSliceNo(&h_slice_param_cuda, i);
			getYver2((uint16_t*)(working_buffer + cb_offset + (i*(MAX_SLICE_DATA ))), h_slice_param_cuda.cb_data, mb_x, mb_y,h_slice_param_cuda.slice_size_in_mb, h_slice_param_cuda.horizontal, h_slice_param_cuda.vertical);
		}
	} else {
		for (i = 0; i<slice_num_max;i++) {
			int mb_x = mbXFormSliceNo(&h_slice_param_cuda, i);
			int mb_y = mbYFormSliceNo(&h_slice_param_cuda, i);
			getCver2((uint16_t*)(working_buffer + cb_offset + (i*(MAX_SLICE_DATA ))), h_slice_param_cuda.cb_data, mb_x, mb_y,h_slice_param_cuda.slice_size_in_mb, h_slice_param_cuda.horizontal, h_slice_param_cuda.vertical);
		}
	}

	if (h_slice_param_cuda.format_444 == true) {
		for (i = 0; i<slice_num_max;i++) {
			int mb_x = mbXFormSliceNo(&h_slice_param_cuda, i);
			int mb_y = mbYFormSliceNo(&h_slice_param_cuda, i);
			getYver2((uint16_t*)(working_buffer + cr_offset + (i*(MAX_SLICE_DATA ))), h_slice_param_cuda.cr_data, mb_x, mb_y,h_slice_param_cuda.slice_size_in_mb, h_slice_param_cuda.horizontal, h_slice_param_cuda.vertical);
		}
	} else {
		for (i = 0; i<slice_num_max;i++) {
			int mb_x = mbXFormSliceNo(&h_slice_param_cuda, i);
			int mb_y = mbYFormSliceNo(&h_slice_param_cuda, i);
			getCver2((uint16_t*)(working_buffer + cr_offset + (i*(MAX_SLICE_DATA ))), h_slice_param_cuda.cr_data, mb_x, mb_y,h_slice_param_cuda.slice_size_in_mb, h_slice_param_cuda.horizontal, h_slice_param_cuda.vertical);
		}
	}

#endif
#ifdef CUDA_ENCODER
	cudaError_t err;
	int16_t *c_working_buffer;
//	printf("host %x\n", working_buffer[0]);
	err = cudaMallocAndCpy((void**)&c_working_buffer, working_buffer, working_buffer_size);
	if (err != cudaSuccess) {
		printf("%d\n", __LINE__);
	}
	uint8_t *c_matrix;
	int matrix_size = 64*3;
	err = cudaMalloc(&c_matrix, matrix_size);
	if (err != cudaSuccess) {
		printf("%d\n", __LINE__);
	}
	err = cudaMemcpy((c_matrix),h_slice_param_cuda.luma_matrix, 64, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%d\n", __LINE__);
	}
	err = cudaMemcpy((c_matrix + 64),h_slice_param_cuda.chroma_matrix, 64, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%d\n", __LINE__);
	}
	err = cudaMemcpy((c_matrix+128),h_slice_param_cuda.chroma_matrix, 64, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%d\n", __LINE__);
	}
	uint32_t mb_size[3];
	if (h_slice_param_cuda.format_444 == true) {
		mb_size[0] = 4;
		mb_size[1] = 4;
		mb_size[2] = 4;
	} else {
		mb_size[0] = 4;
		mb_size[1] = 2;
		mb_size[2] = 2;
	}
	uint32_t *c_mb_size;
	err = cudaMallocAndCpy((void**)&c_mb_size, mb_size, 12);
	if (err != cudaSuccess) {
		printf("%d\n", __LINE__);
	}
	printf("sm %d\n", slice_num_max);

	double *c_kc_value;
	int kc_value_size = sizeof(double) * KC_INDEX_MAX;
	err = cudaMallocAndCpy((void**)&c_kc_value, h_kc_value, kc_value_size);
	if (err != cudaSuccess) {
		printf("%d\n", __LINE__);
	}

	uint8_t *c_qscale_table;
	int qscale_table_size = slice_num_max;
	err = cudaMallocAndCpy((void**)&c_qscale_table, h_slice_param_cuda.qscale_table, qscale_table_size);
	if (err != cudaSuccess) {
		printf("%d\n", __LINE__);
	}
	int nElem = slice_num_max*3;
	dim3 block(32, 1);
	dim3 grid((nElem + block.x -1 / block.x) );

	double iStart = cpuSecond();
	dct_and_quant<<<grid,block>>>(c_working_buffer, c_matrix,  h_slice_param_cuda.slice_size_in_mb,  c_mb_size, c_kc_value, c_qscale_table , slice_num_max);
	CHECK(cudaDeviceSynchronize());
	err = cudaMemcpy(working_buffer, c_working_buffer, working_buffer_size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("%d\n", __LINE__);
	}
	double iElaps = cpuSecond() - iStart;

	printf("time = %f\n", iElaps);

#else
	uint8_t *matrix = (uint8_t*)malloc(64*3);
	if (matrix == NULL) {
		printf("%d\n", __LINE__);
	}
	memcpy(matrix, h_slice_param_cuda.luma_matrix, 64);
	memcpy(matrix+64, h_slice_param_cuda.chroma_matrix, 64);
	memcpy(matrix+128, h_slice_param_cuda.chroma_matrix, 64);
	uint32_t *mb_size =  (uint32_t*)malloc(4*3);
	if (mb_size == NULL) {
		printf("%d\n", __LINE__);
	}
	if (h_slice_param_cuda.format_444 == true) {
		mb_size[0] = MB_IN_BLOCK;
		mb_size[1] = MB_IN_BLOCK;
		mb_size[2] = MB_IN_BLOCK;
	} else {
		mb_size[0] = MB_IN_BLOCK;
		mb_size[1] = MB_422C_IN_BLCCK;
		mb_size[2] = MB_422C_IN_BLCCK;
	}
	for (i = 0; i< slice_num_max*3;i++)  {
		dct_and_quant(i,working_buffer , matrix, h_slice_param_cuda.slice_size_in_mb, mb_size, h_kc_value, h_slice_param_cuda.qscale_table, slice_num_max);
	}

#endif

	for (i = 0; i< slice_num_max;i++)  {
		encode_slices2(&h_slice_param_cuda, i, slice_size_table, write_bitstream, working_buffer, h_kc_value);
	}


    for (i = 0; i < slice_num_max ; i++) {
        setSliceTalbeFlush(slice_size_table[i], slice_size_table_offset + (i * 2));
    }

}

uint8_t *encode_frame(struct encoder_param* param, uint32_t *encode_frame_size)
{

	write_bitstream = (struct bitstream*)malloc(sizeof(struct bitstream) + MAX_BITSTREAM_SIZE);
	if (write_bitstream == NULL ) {
		printf("error malloc %d\n", __LINE__);
		return NULL;
	}
//	write_bitstream.bitstream_buffer = bitstream_buffer;
    initBitStream(write_bitstream);

    uint32_t frame_size_offset = getBitSize(write_bitstream) >> 3 ;
    uint32_t frame_size = SET_DATA32(0x0); 
    setByte(write_bitstream, (uint8_t*)&frame_size,4);

    uint32_t frame_identifier = SET_DATA32(0x69637066); //icpf


    setByte(write_bitstream, (uint8_t*)&frame_identifier,4);

    set_frame_header(param);
    uint32_t picture_size_offset = (getBitSize(write_bitstream)) >> 3 ;

    set_picture_header(param);

    encode_slices(param);
    uint32_t picture_end = (getBitSize(write_bitstream)) >>  3 ;

    uint32_t tmp  = picture_end - picture_size_offset;
    uint32_t picture_size = SET_DATA32(tmp);

    setByteInOffset(write_bitstream, picture_size_offset_, (uint8_t*)&picture_size, 4);


    uint8_t *ptr = getBitStream(write_bitstream, encode_frame_size);
    uint32_t frame_size_data = SET_DATA32(*encode_frame_size);
    setByteInOffset(write_bitstream, frame_size_offset, (uint8_t*)&frame_size_data , 4);
    return ptr;
}




void encoder_init(void)
{
	//vlc_init();
	dct_init(h_kc_value);
}




