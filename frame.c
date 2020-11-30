/**
 *
 * Copyright (c) 2020 Yuusuke Miyazaki
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 *
 **/

#define _GNU_SOURCE 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <pthread.h>

//#include <linux/fcntl.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


#include "config.h"
#include "dct.h"
#include "bitstream.h"
#include "encoder.h"
#include "slice.h"

void setSliceTalbeFlush(uint16_t size, uint32_t offset) {
    uint16_t slice_size = SET_DATA16(size);
    setByteInOffset(offset, (uint8_t*)&slice_size, 2);
    

}
#define MACRO_BLOCK_Y_HORIZONTAL  (16)
#define MACRO_BLOCK_Y_VERTICAL    (16)

#define MACRO_BLOCK_422_C_HORIZONTAL  (8)
#define MACRO_BLOCK_422_C_VERTICAL    (16)



// MACRO_BLOCK_Y_HORIZONTAL * MACRO_BLOCK_Y_VERTICAL * sizeof(uint16_t) = 512
//mb_size * 512  
#define MAX_SLICE_DATA_SIZE		(8*512)
uint8_t y_slice_data[MAX_SLICE_DATA_SIZE];
uint8_t cb_slice_data[MAX_SLICE_DATA_SIZE];
uint8_t cr_slice_data[MAX_SLICE_DATA_SIZE];

/* get data for one slice */
void  getY(uint16_t *out, uint16_t *in, uint32_t mb_x, uint32_t mb_y, int32_t mb_size, int32_t horizontal, int32_t vertical)
{
    for(int32_t i = 0;i<MACRO_BLOCK_Y_VERTICAL;i++) {
        memcpy(out + i * (mb_size * MACRO_BLOCK_Y_HORIZONTAL), 
               in + (mb_x * MACRO_BLOCK_Y_HORIZONTAL) + ((mb_y * MACRO_BLOCK_Y_VERTICAL) * horizontal) + (i * horizontal), 
				//MACRO_BLOCK_Y_HORIZONTAL * sizeof(uint16_t) = 32
               mb_size * 32);
    }
    return ;

}
/* get data for one slice */
/* for 422 */
void getC(uint16_t *out, uint16_t *in, uint32_t mb_x, uint32_t mb_y, int32_t mb_size, int32_t horizontal, int32_t vertical)
{

    for(int32_t i = 0;i<MACRO_BLOCK_422_C_VERTICAL;i++) {
        memcpy(out + i * (mb_size * MACRO_BLOCK_422_C_HORIZONTAL), 
               in + (mb_x * MACRO_BLOCK_422_C_HORIZONTAL) + ((mb_y * MACRO_BLOCK_422_C_VERTICAL) * (horizontal/2)) + (i * (horizontal/2)), 
				//MACRO_BLOCK_422_C_HORIZONTAL * sizeof(uint16_t) = 16
               mb_size * 16);

    }
    return;

}
void start_write_bitstream(void);
void frame_end_wait(void);

uint16_t slice_size_table[MAX_SLICE_NUM];

uint8_t slice_bitstream[MAX_THREAD_NUM];

struct Slice slice_param[MAX_SLICE_NUM];
//extern void thread_start(void);
//extern void thread_end(void);

int thread_fd[MAX_THREAD_NUM];

uint32_t slice_num_max;
void encode_slices(struct encoder_param * param)
{
    uint32_t mb_x;
    uint32_t mb_y;
    uint32_t mb_x_max;
    mb_x_max = (param->horizontal+ 15 ) >> 4;
//	uint32_t thread_num;


    slice_num_max = GetSliceNum(param->horizontal, param->vertical, param->slice_size_in_mb);

    int32_t slice_mb_count = param->slice_size_in_mb;
    mb_x = 0;
    mb_y = 0;

    /* write dummy slice size table */
    int32_t i;
    uint32_t slice_size_table_offset = (getBitSize()) >> 3 ;
    for (i = 0; i < slice_num_max ; i++) {
        uint16_t slice_size = 0x0;
        setByte((uint8_t*)&slice_size, 2);
    }

    slice_mb_count = param->slice_size_in_mb;
    mb_x = 0;
    mb_y = 0;
//	thread_num = 0;
    for (i = 0; i < slice_num_max ; i++) {

        while ((mb_x_max - mb_x) < slice_mb_count)
            slice_mb_count >>=1;

       //printf("%d %d\n", mb_x, mb_y);
//       uint32_t size;
#if 0
        getY((uint16_t*)y_slice_data, param->y_data, mb_x,mb_y,slice_mb_count, param->horizontal, param->vertical);
        if (param->format_444 ==  true) {
            getY((uint16_t*)cb_slice_data, param->cb_data, mb_x,mb_y,slice_mb_count, param->horizontal, param->vertical);
            getY((uint16_t*)cr_slice_data, param->cr_data, mb_x,mb_y,slice_mb_count, param->horizontal, param->vertical);
       } else {
            getC((uint16_t*)cb_slice_data, param->cb_data, mb_x,mb_y,slice_mb_count, param->horizontal, param->vertical);
            getC((uint16_t*)cr_slice_data, param->cr_data, mb_x,mb_y,slice_mb_count, param->horizontal, param->vertical);
       }
#endif

//       struct Slice slice_param;
//	   slice_param[i].thread_num = thread_num;
       slice_param[i].slice_no = i;
       slice_param[i].luma_matrix = param->luma_matrix;
       slice_param[i].chroma_matrix = param->chroma_matrix;
       slice_param[i].qscale = param->qscale_table[i];
       slice_param[i].slice_size_in_mb= param->slice_size_in_mb;
       slice_param[i].horizontal= param->horizontal;
       slice_param[i].vertical= param->vertical;
#if 0
       slice_param.y_data= (uint16_t*)y_slice_data;
       slice_param.cb_data= (uint16_t*)cb_slice_data;
       slice_param.cr_data= (uint16_t*)cr_slice_data;
       slice_param.mb_x = 0;
       slice_param.mb_y = 0;
#else
       slice_param[i].y_data= (uint16_t*)param->y_data;
       slice_param[i].cb_data= (uint16_t*)param->cb_data;
       slice_param[i].cr_data= (uint16_t*)param->cr_data;
       slice_param[i].mb_x = mb_x;
       slice_param[i].mb_y = mb_y;
#endif
       slice_param[i].format_444 = param->format_444;

	   if (i == (slice_num_max -1)) {
			slice_param[i].end = 1;
		} else {
			slice_param[i].end = 0;
		}

		printf("write start %d\n", i);
		struct Slice *ptr = &slice_param[i];
		write(thread_fd[i%MAX_THREAD_NUM], &ptr, sizeof(struct Slice*));
		printf("write end\n");

       //size = encode_slice(y_data, cb_data, cr_data, mb_x, mb_y, slice_size);
       /* need mb_x = 0 and mb_y = 0 becase getY and getC takas data to mb_x=0 and mb_y=0 position . */
//       size = encode_slice(&slice_param);
//       slice_size_table[i] = size;
       //printf("size = %d\n",size);

        mb_x += slice_mb_count;
        if (mb_x == mb_x_max ) {
            slice_mb_count = param->slice_size_in_mb;
            mb_x = 0;
            mb_y++;
        }
		
    }
	
//       size = encode_slice(&slice_param);
//	thread_start();
//	thread_end();
//	printf("thread end\n");
//       slice_size_table[i] = size;

	printf("start_write_bitstream\n");
	start_write_bitstream();

	printf("wait threads\n");
	frame_end_wait();

    for (i = 0; i < slice_num_max ; i++) {
        setSliceTalbeFlush(slice_size_table[i], slice_size_table_offset + (i * 2));
    }

}

void write_slice_size(int slice_no, int size)
{
	slice_size_table[slice_no] = size;
	return;
}
uint32_t picture_size_offset_ = 0;
void set_picture_header(struct encoder_param* param)
{

    uint8_t picture_header_size = 0x8;
    setBit(picture_header_size, 5);

    uint8_t reserved = 0x0;
    setBit(reserved , 3);

    picture_size_offset_ = (getBitSize()) >> 3 ;

    uint32_t picture_size = SET_DATA32(0);
    setByte((uint8_t*)&picture_size, 4);

    uint32_t slice_num = GetSliceNum(param->horizontal, param->vertical, param->slice_size_in_mb);
    uint16_t deprecated_number_of_slices =  SET_DATA16(slice_num);
    setByte((uint8_t*)&deprecated_number_of_slices , 0x2);


    uint8_t reserved2 = 0x0;
    setBit(reserved2 , 2);

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
    setBit(log2_desired_slice_size_in_mb, 2);

    uint8_t reserved3 = 0x0;
    setBit(reserved3 , 4);


}
void set_frame_header(struct encoder_param* param)
{
    uint16_t frame_header_size = SET_DATA16(0x94);
    setByte((uint8_t*)&frame_header_size, 0x2);

    uint8_t reserved = 0x0;
    setByte(&reserved, 0x1);

    uint8_t bitstream_version = 0x0;
    setByte(&bitstream_version, 0x1);


    uint32_t encoder_identifier = SET_DATA32(0x4c617663);
    setByte((uint8_t*)&encoder_identifier, 0x4);

    uint16_t horizontal_size = SET_DATA16(param->horizontal);
    setByte((uint8_t*)&horizontal_size , 0x2);

    uint16_t vertical_size = SET_DATA16(param->vertical);
    setByte((uint8_t*)&vertical_size, 0x2);


    uint8_t chroma_format;
    if (param->format_444 == true) {
        chroma_format = 0x3;
    } else {
        chroma_format = 0x2;
    }
    setBit(chroma_format, 2);

    uint8_t reserved1 = 0x0;
    setBit(reserved1, 2);

    uint8_t interlace_mode = 0;
    setBit(interlace_mode, 2);

    uint8_t reserved2 = 0x0;
    setBit(reserved2, 2);

    uint8_t aspect_ratio_information = 0;
    setBit(aspect_ratio_information, 4);

    uint8_t frame_rate_code = 0;
    setBit(frame_rate_code, 4);

    uint8_t color_primaries = 0x0;
    setByte(&color_primaries, 1);

    uint8_t transfer_characteristic = 0x0;
    setByte(&transfer_characteristic , 1);

    uint8_t matrix_coefficients = 0x2;
    setByte(&matrix_coefficients, 1);


    uint8_t reserved3 = 0x4;
    setBit(reserved3 , 4);

    //printf("1   %x %x\n", tmp_buf_byte_offset, tmp_buf[0x1b]);
    uint8_t alpha_channel_type = 0x0;
    setBit(alpha_channel_type , 4);

    //printf("2   %x %x\n", tmp_buf_byte_offset, tmp_buf[0x1b]);
    uint8_t reserved4 = 0x0;
    setByte(&reserved4 , 1);
    
    //printf("3   %x %x\n", tmp_buf_byte_offset, tmp_buf[0x1b]);
    uint8_t reserved5 = 0x0;
    setBit(reserved5, 6);
    
    //printf("4   %x %x\n", tmp_buf_byte_offset, tmp_buf[0x1b]);
    uint8_t load_luma_quantization_matrix = 0x1;
    setBit(load_luma_quantization_matrix, 1);

    //printf("5   %x %x\n", tmp_buf_byte_offset, tmp_buf[0x1b]);

    uint8_t load_chroma_quantization_matrix = 0x1;
    setBit(load_chroma_quantization_matrix, 1);

    setByte(param->luma_matrix, MATRIX_NUM );
    setByte(param->chroma_matrix, MATRIX_NUM );


}
uint8_t *encode_frame(struct encoder_param* param, uint32_t *encode_frame_size)
{

    initBitStream();

    uint32_t frame_size_offset = getBitSize() >> 3 ;
    uint32_t frame_size = SET_DATA32(0x0); 
    setByte((uint8_t*)&frame_size,4);

    uint32_t frame_identifier = SET_DATA32(0x69637066); //icpf


    setByte((uint8_t*)&frame_identifier,4);

    set_frame_header(param);
    uint32_t picture_size_offset = (getBitSize()) >> 3 ;

    set_picture_header(param);

    encode_slices(param);
    uint32_t picture_end = (getBitSize()) >>  3 ;

    uint32_t tmp  = picture_end - picture_size_offset;
    //printf("%x\n", tmp);
    uint32_t picture_size = SET_DATA32(tmp);
    //printf("picture_size2 = %x\n", picture_size);
    //for debug
    setByteInOffset(picture_size_offset_, (uint8_t*)&picture_size, 4);
    //printf("picture_size1 = %x\n", picture_end - picture_size_offset);
    //printf("picture_size2 = %x\n", picture_size);


    uint8_t *ptr = getBitStream(encode_frame_size);
    uint32_t frame_size_data = SET_DATA32(*encode_frame_size);
    setByteInOffset(frame_size_offset, (uint8_t*)&frame_size_data , 4);
    return ptr;
}

struct thread_param {
	int seq;
	int thread_no;
//	pthread_mutex_t  my_thread_mutex;
	pthread_mutex_t  write_bitstream_my_mutex;
	pthread_mutex_t  *write_bitstream_next_mutex;
};



void wait_write_bitstream(struct thread_param * param)
{
//	printf("%d %p\n", param->seq, &param->write_bitstream_next_mutex);
	int ret = pthread_mutex_lock(&param->write_bitstream_my_mutex);
	if (ret != 0) {
		printf("%d\n", __LINE__);
		return;
	}
	return;
}
void start_write_next_bitstream(struct thread_param * param)
{
//	printf("%d %p\n", param->seq, param->write_bitstream_next_mutex);
	if (param->write_bitstream_next_mutex != NULL ) {
		int ret = pthread_mutex_unlock(param->write_bitstream_next_mutex);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return;
		}
	}
	return;
}

//pthread_mutex_t start_frame_mutex;
//pthread_cond_t start_frame_cond;
//int start_frame = 0;

pthread_mutex_t end_frame_mutex;


void *thread_start_routin(void *arg)
{
	struct thread_param *param =  (struct thread_param*)arg;

	char fifoname[1024];
	sprintf(fifoname, "/tmp/fifo%d", param->thread_no);	
	mkfifo(fifoname, 0666);
	int fd = open(fifoname, O_RDONLY);

	for(;;) {
		struct Slice *slice;
		read(fd, &slice, sizeof(slice));

		//start frame encode
		int counter = 0;
//		int seq = (counter * MAX_THREAD_NUM) + param->thread_no;

//		printf("seq %d\n", seq);
		encode_slice(slice);

		printf("wait_write_bitstream\n");
		wait_write_bitstream(param);
		printf("write_bitstream\n");
		if (slice->end == true) {
			printf("end of frame\n");
			pthread_mutex_unlock(&end_frame_mutex);
		} else {
			start_write_next_bitstream(param);
		}

		counter++;

	}
	
}

pthread_t thread[MAX_THREAD_NUM];

struct thread_param params[MAX_THREAD_NUM];


void frame_end_mutex_init(void) {
	int ret;
	ret = pthread_mutex_init(&end_frame_mutex,NULL);
	if (ret != 0) {
		printf("%d\n", __LINE__);
		return;
	}
	pthread_mutex_lock(&end_frame_mutex);
}
void frame_end_wait(void) {	
	pthread_mutex_lock(&end_frame_mutex);
}


int encoder_thread_init(void)
{
	int i,ret;

	for(i=0;i<MAX_THREAD_NUM;i++) {
		ret = pthread_mutex_init(&params[i].write_bitstream_my_mutex,NULL);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return -1;
		}

		ret = pthread_mutex_lock(&params[i].write_bitstream_my_mutex);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return -1;
		}
	}
//	printf("%d\n", __LINE__);
	for(i=0;i<MAX_THREAD_NUM;i++) {
		params[i].write_bitstream_next_mutex = &params[(i+1)%MAX_THREAD_NUM].write_bitstream_my_mutex;
	}
//	printf("%d\n", __LINE__);

	pthread_attr_t attr;
	ret = pthread_attr_init(&attr);
	if (ret != 0) {
		printf("%d\n", __LINE__);
		return -1;
	}
	for(i=0;i<MAX_THREAD_NUM;i++) {
		params[i].thread_no = i;

		ret = pthread_create(&thread[i], &attr, &thread_start_routin, (void*)&params[i]);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return -1;
		}
	}
	for(i=0;i<MAX_THREAD_NUM;i++) {
		params[i].thread_no = i;
	}

	for(i=0;i<MAX_THREAD_NUM;i++) {
		char fifoname[1024];
		sprintf(fifoname, "/tmp/fifo%d", i);	
		mkfifo(fifoname, 0666);
		thread_fd[i] = open(fifoname, O_WRONLY);
		fcntl(thread_fd[i], F_SETPIPE_SZ, (20971520/ MAX_THREAD_NUM));
	}
	frame_end_mutex_init();

	return 0;
}

void start_write_bitstream(void) {
		pthread_mutex_unlock(&params[0].write_bitstream_my_mutex);
}


void encoder_init(void)
{
    int32_t i,j;
    for(i=0;i<MATRIX_NUM ;i++) {
        for(j=0;j<MATRIX_NUM ;j++) {
            if (i == block_pattern_scan_table[j]  ) {
                block_pattern_scan_read_order_table[i] = j;
                break;
            }
        }
    }
#ifdef PRE_CALC_COS
	dct_init();
#endif
	int ret = encoder_thread_init();
	if (ret != 0) {
		printf("%d", __LINE__);
		return;
	}

}




int32_t GetSliceNum(int32_t horizontal, int32_t vertical, int32_t sliceSize)
{
    int32_t mb_x_max = (horizontal + 15)  >> 4;
    int32_t mb_y_max = (vertical + 15) >> 4;


    int32_t slice_num_max;

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

    slice_num_max = number_of_slices_per_mb_row * mb_y_max;
    return slice_num_max;

}
uint32_t GetEncodeHorizontal(int32_t horizontal)
{
    return ((horizontal + 15)  >> 4) << 4;

}
uint32_t GetEncodeVertical(int32_t vertical)
{
    return ((vertical + 15)  >> 4) << 4;
}

