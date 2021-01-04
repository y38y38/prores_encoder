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
#include "bitstream.h"
#include "vlc.h"
#include "slice.h"
#include "encoder.h"



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

	memcpy(h_slice_param_cuda.luma_matrix, param->luma_matrix, BLOCK_IN_PIXEL);
    memcpy(h_slice_param_cuda.chroma_matrix, param->chroma_matrix, BLOCK_IN_PIXEL);
    h_slice_param_cuda.slice_size_in_mb= param->slice_size_in_mb;
    h_slice_param_cuda.horizontal= param->horizontal;
    h_slice_param_cuda.vertical= param->vertical;
    h_slice_param_cuda.format_444 = param->format_444;
	h_slice_param_cuda.slice_num_max = slice_num_max;
	struct Slice_cuda * c_slice_param_cuda;

#ifdef CUDA_ENCODER
	cudaError_t err;
	err = cudaMalloc(&c_slice_param_cuda, sizeof(struct Slice_cuda));
	if (err != cudaSuccess) {
		printf("cudaMemcpy error %d %d", __LINE__, err);
	}
	err = cudaMemcpy(c_slice_param_cuda, &h_slice_param_cuda, sizeof(struct Slice_cuda), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("cudaMemcpy error %d %d", __LINE__, err);
	}
#else
	c_slice_param_cuda = (struct Slice_cuda *)malloc(sizeof(struct Slice_cuda));
	if (c_slice_param_cuda == NULL ) {
		printf("malloc error %d", __LINE__);
	}
	memcpy(c_slice_param_cuda, &h_slice_param_cuda, sizeof(struct Slice_cuda));

#endif

	uint8_t *c_qscale_table;
#ifdef CUDA_ENCODER
	err = cudaMalloc(&c_qscale_table, sizeof(uint8_t) * slice_num_max);
	if (err != cudaSuccess) {
		printf("cudaMemcpy error %d %d", __LINE__, err);
	}
	err = cudaMemcpy(c_qscale_table, param->qscale_table, slice_num_max, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("cudaMemcpy error %d %d", __LINE__, err);
	}
#else
	c_qscale_table = (uint8_t*)malloc(sizeof(uint8_t) * slice_num_max);
	if (c_qscale_table == NULL ) {
		printf("malloc error %d", __LINE__);
	}
	memcpy(c_qscale_table, param->qscale_table, slice_num_max);
#endif

	uint16_t *c_y_data;
	int y_size = param->horizontal * param->vertical * sizeof(uint16_t);
#ifdef CUDA_ENCODER

	err = cudaMalloc(&c_y_data, y_size);
	if (err != cudaSuccess) {
		printf("cudaMemcpy error %d %d", __LINE__, err);
	}
	err = cudaMemcpy(c_y_data, param->y_data, y_size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("cudaMemcpy error %d %d", __LINE__, err);
	}
#else
	c_y_data = (uint16_t*)malloc(y_size);
	if (c_y_data == NULL ) {
		printf("malloc error %d", __LINE__);
	}
	memcpy(c_y_data, param->y_data, y_size);
#endif

	uint16_t *c_cb_data;
	int cb_size;
	if (param->format_444 == true) {
		cb_size = y_size;
	} else {
		cb_size = y_size >> 1;
	}

#ifdef CUDA_ENCODER
	err = cudaMalloc(&c_cb_data, cb_size);
	if (err != cudaSuccess) {
		printf("cudaMemcpy error %d %d", __LINE__, err);
	}
	err = cudaMemcpy(c_cb_data, param->cb_data, cb_size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("cudaMemcpy error %d %d", __LINE__, err);
	}
#else
	c_cb_data = (uint16_t*)malloc(cb_size);
	if (c_cb_data == NULL ) {
		printf("malloc error %d", __LINE__);
	}
	memcpy(c_cb_data, param->cb_data, cb_size);
#endif

	uint16_t *c_cr_data;
	int cr_size = cb_size;
#ifdef CUDA_ENCODER
	err = cudaMalloc(&c_cr_data, cr_size);
	if (err != cudaSuccess) {
		printf("cudaMemcpy error %d %d", __LINE__, err);
	}
	err = cudaMemcpy(c_cr_data, param->cb_data, cr_size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("cudaMemcpy error %d %d", __LINE__, err);
	}
#else
	c_cr_data = (uint16_t*)malloc(cr_size);
	if (c_cr_data == NULL ) {
		printf("malloc error %d", __LINE__);
	}
	memcpy(c_cr_data, param->cr_data, cr_size);
#endif

	struct bitstream *c_bitstream;
	int bitstream_size = (sizeof(struct bitstream) + MAX_SLICE_BITSTREAM_SIZE) * slice_num_max;
#ifdef CUDA_ENCODER
	err = cudaMalloc(&c_bitstream, bitstream_size);
	if (err != cudaSuccess) {
		printf("cudaMemcpy error %d %d", __LINE__, err);
	}
#else
	//printf("malloc size%d\n", bitstream_size);
	c_bitstream = (struct bitstream*)malloc(bitstream_size);
	if (c_bitstream == NULL ) {
		printf("malloc error %d", __LINE__);
	}
	memset(c_bitstream, 0x0, bitstream_size);
#endif


	uint16_t *c_slice_size_table;
	int slice_size_table_size = slice_num_max * sizeof(uint16_t);
#ifdef CUDA_ENCODER
	err = cudaMalloc(&c_slice_size_table, slice_size_table_size);
	if (err != cudaSuccess) {
		printf("cudaMemcpy error %d %d", __LINE__, err);
	}
#else
	c_slice_size_table = (uint16_t*)malloc(slice_size_table_size);
	if (c_slice_size_table == NULL ) {
		printf("malloc error %d", __LINE__);
	}

#endif

	int16_t *c_working_buffer;//thread分のバッファを持つ必要あり。
	int working_buffer_size = (MAX_SLICE_DATA * 2) * slice_num_max;
#ifdef CUDA_ENCODER
	err = cudaMalloc(&c_working_buffer, working_buffer_size);
	if (err != cudaSuccess) {
		printf("cudaMemcpy error %d %d", __LINE__, err);
	}
#else
	c_working_buffer = (int16_t*)malloc(working_buffer_size);
	if (c_working_buffer == NULL ) {
		printf("malloc error %d", __LINE__);
	}

#endif


	double *c_kc_value;
	int kc_value_size = sizeof(double) * KC_INDEX_MAX;
#ifdef CUDA_ENCODER
	err = cudaMalloc(&c_kc_value, kc_value_size);
	if (err != cudaSuccess) {
		printf("cudaMalloc error %d %d", __LINE__, err);
	}
	err = cudaMemcpy(c_kc_value, h_kc_value, kc_value_size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("cudaMalloc error %d %d", __LINE__, err);
	}
#else
	c_kc_value = (double*)malloc(kc_value_size);
	if (c_kc_value == NULL ) {
		printf("cudaMalloc error %d", __LINE__);
	}
	memcpy(c_kc_value, h_kc_value, kc_value_size);
#endif	

#ifdef  CUDA_ENCODER
	i = 0;
#if 1
	int nElem = slice_num_max;
	dim3 block(1, 1);
	dim3 grid(slice_num_max);
	encode_slice<<<grid,block>>>(i, c_slice_param_cuda, c_qscale_table, c_y_data, c_cb_data, c_cr_data, c_bitstream, c_slice_size_table, c_working_buffer,c_kc_value);
#else
	encode_slice<<<1,1>>>(i, c_slice_param_cuda, c_qscale_table, c_y_data, c_cb_data, c_cr_data, c_bitstream, c_slice_size_table, c_working_buffer,c_kc_value);
#endif
#else
	//int i;
	for(i = 0; i <slice_num_max;i++)  {
		encode_slice(i, c_slice_param_cuda, c_qscale_table, c_y_data, c_cb_data, c_cr_data, c_bitstream, c_slice_size_table, c_working_buffer,c_kc_value);
	}
#endif



#ifdef CUDA_ENCODER
	err = cudaMemcpy(slice_size_table, c_slice_size_table, slice_size_table_size,cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("cudaMalloc error %d %d", __LINE__, err);
	}
	//printf("slice table %x\n", slice_size_table[0]);
#else
	memcpy(slice_size_table, c_slice_size_table, slice_size_table_size);
#endif

	struct bitstream* h_bitstream;
#ifdef CUDA_ENCODER
	h_bitstream = (struct bitstream*)malloc(bitstream_size);
	if (h_bitstream == NULL ) {
		printf("malloc error %d", __LINE__);
	}
	err = cudaMemcpy(h_bitstream, c_bitstream, bitstream_size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("cudaMalloc error %d %d", __LINE__, err);
	}
#else
	h_bitstream = (struct bitstream*)malloc(bitstream_size);
	if (h_bitstream == NULL ) {
		printf("malloc error %d", __LINE__);
	}
	memcpy(h_bitstream, c_bitstream, bitstream_size);
#endif

    for (i = 0; i < slice_num_max ; i++) {
		//printf("size=0x%x %x\n", slice_size_table[i], slice_size_table_size);
        setSliceTalbeFlush(slice_size_table[i], slice_size_table_offset + (i * 2));
		for(int j=0;j<128;j++) {
			//printf("%x ", c_bitstream[i].bitstream_buffer[j]);
		}
		//printf("\n%x\n", c_bitstream[i].bitstream_buffer);
		uint8_t *ptr = (uint8_t*)h_bitstream;
		struct bitstream * bptr = (struct bitstream*)(ptr + ((sizeof(struct bitstream) + MAX_SLICE_BITSTREAM_SIZE) * i));

		setByte(write_bitstream, bptr->bitstream_buffer, slice_size_table[i]);
    }
#ifdef CUDA_ENCODER
	cudaFree(c_slice_param_cuda);
	cudaFree(c_qscale_table);
	cudaFree(c_y_data);
	cudaFree(c_cb_data);
	cudaFree(c_cr_data);
	cudaFree(c_bitstream);
	cudaFree(c_slice_size_table);
	cudaFree(c_working_buffer);
	cudaFree(c_kc_value);
#else
	free(c_slice_param_cuda);
	free(c_qscale_table);
	free(c_y_data);
	free(c_cb_data);
	free(c_cr_data);
	free(c_bitstream);
	free(c_slice_size_table);
	free(c_working_buffer);
	free(c_kc_value);
#endif



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




