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


#define MAX_SLICE_DATA	(2048)

struct thread_param {
	int thread_no;
	pthread_mutex_t  write_bitstream_my_mutex;
	pthread_mutex_t  *write_bitstream_next_mutex;
	int16_t y_slice[MAX_SLICE_DATA];
};



#define MACRO_BLOCK_Y_HORIZONTAL  (16)
#define MACRO_BLOCK_Y_VERTICAL    (16)

#define MACRO_BLOCK_422_C_HORIZONTAL  (8)
#define MACRO_BLOCK_422_C_VERTICAL    (16)

#define MAX_BITSTREAM_SIZE	(1073741824) //1M



static struct bitstream write_bitstream;
static uint8_t bitstream_buffer[MAX_BITSTREAM_SIZE];



//void start_write_bitstream(void);
//void frame_end_wait(void);

static uint16_t slice_size_table[MAX_SLICE_NUM];

static struct bitstream slice_bitstream[MAX_THREAD_NUM];
static uint8_t slice_bistream_buffer[MAX_THREAD_NUM][MAX_SLICE_BITSTREAM_SIZE];


static struct Slice slice_param[MAX_SLICE_NUM];

static pthread_mutex_t slice_num_thread_mutex[MAX_THREAD_NUM];
static pthread_cond_t slice_num_thread_cond[MAX_THREAD_NUM];
static int slice_num_thread[MAX_THREAD_NUM];

static struct thread_param params[MAX_THREAD_NUM];

pthread_mutex_t end_frame_mutex;

static pthread_t thread[MAX_THREAD_NUM];


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
    setBit(&write_bitstream, picture_header_size, 5);

    uint8_t reserved = 0x0;
    setBit(&write_bitstream, reserved , 3);

    picture_size_offset_ = (getBitSize(&write_bitstream)) >> 3 ;

    uint32_t picture_size = SET_DATA32(0);
    setByte(&write_bitstream, (uint8_t*)&picture_size, 4);

    uint32_t slice_num = GetSliceNum(param->horizontal, param->vertical, param->slice_size_in_mb);
    uint16_t deprecated_number_of_slices =  SET_DATA16(slice_num);
    setByte(&write_bitstream, (uint8_t*)&deprecated_number_of_slices , 0x2);


    uint8_t reserved2 = 0x0;
    setBit(&write_bitstream, reserved2 , 2);

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
    setBit(&write_bitstream,log2_desired_slice_size_in_mb, 2);

    uint8_t reserved3 = 0x0;
    setBit(&write_bitstream, reserved3 , 4);


}
void set_frame_header(struct encoder_param* param)
{
    uint16_t frame_header_size = SET_DATA16(0x94);
    setByte(&write_bitstream, (uint8_t*)&frame_header_size, 0x2);

    uint8_t reserved = 0x0;
    setByte(&write_bitstream, &reserved, 0x1);

    uint8_t bitstream_version = 0x0;
    setByte(&write_bitstream, &bitstream_version, 0x1);


    uint32_t encoder_identifier = SET_DATA32(0x4c617663);
    setByte(&write_bitstream, (uint8_t*)&encoder_identifier, 0x4);

    uint16_t horizontal_size = SET_DATA16(param->horizontal);
    setByte(&write_bitstream, (uint8_t*)&horizontal_size , 0x2);

    uint16_t vertical_size = SET_DATA16(param->vertical);
    setByte(&write_bitstream, (uint8_t*)&vertical_size, 0x2);


    uint8_t chroma_format;
    if (param->format_444 == true) {
        chroma_format = 0x3;
    } else {
        chroma_format = 0x2;
    }
    setBit(&write_bitstream, chroma_format, 2);

    uint8_t reserved1 = 0x0;
    setBit(&write_bitstream, reserved1, 2);

    uint8_t interlace_mode = 0;
    setBit(&write_bitstream, interlace_mode, 2);

    uint8_t reserved2 = 0x0;
    setBit(&write_bitstream, reserved2, 2);

    uint8_t aspect_ratio_information = 0;
    setBit(&write_bitstream, aspect_ratio_information, 4);

    uint8_t frame_rate_code = 0;
    setBit(&write_bitstream, frame_rate_code, 4);

    uint8_t color_primaries = 0x0;
    setByte(&write_bitstream, &color_primaries, 1);

    uint8_t transfer_characteristic = 0x0;
    setByte(&write_bitstream, &transfer_characteristic , 1);

    uint8_t matrix_coefficients = 0x2;
    setByte(&write_bitstream, &matrix_coefficients, 1);


    uint8_t reserved3 = 0x4;
    setBit(&write_bitstream, reserved3 , 4);

    //printf("1   %x %x\n", tmp_buf_byte_offset, tmp_buf[0x1b]);
    uint8_t alpha_channel_type = 0x0;
    setBit(&write_bitstream, alpha_channel_type , 4);

    //printf("2   %x %x\n", tmp_buf_byte_offset, tmp_buf[0x1b]);
    uint8_t reserved4 = 0x0;
    setByte(&write_bitstream, &reserved4 , 1);
    
    //printf("3   %x %x\n", tmp_buf_byte_offset, tmp_buf[0x1b]);
    uint8_t reserved5 = 0x0;
    setBit(&write_bitstream, reserved5, 6);
    
    //printf("4   %x %x\n", tmp_buf_byte_offset, tmp_buf[0x1b]);
    uint8_t load_luma_quantization_matrix = 0x1;
    setBit(&write_bitstream, load_luma_quantization_matrix, 1);

    //printf("5   %x %x\n", tmp_buf_byte_offset, tmp_buf[0x1b]);

    uint8_t load_chroma_quantization_matrix = 0x1;
    setBit(&write_bitstream, load_chroma_quantization_matrix, 1);

    setByte(&write_bitstream, param->luma_matrix, MATRIX_NUM );
    setByte(&write_bitstream, param->chroma_matrix, MATRIX_NUM );


}

void setSliceTalbeFlush(uint16_t size, uint32_t offset) {
    uint16_t slice_size = SET_DATA16(size);
    setByteInOffset(&write_bitstream, offset, (uint8_t*)&slice_size, 2);
    

}
void wait_write_bitstream(struct thread_param * param)
{
	int ret = pthread_mutex_lock(&param->write_bitstream_my_mutex);
	if (ret != 0) {
		printf("%d\n", __LINE__);
		return;
	}
	return;
}
void start_write_next_bitstream(struct thread_param * param)
{
	if (param->write_bitstream_next_mutex != NULL ) {
		int ret = pthread_mutex_unlock(param->write_bitstream_next_mutex);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return;
		}
	}
	return;
}

void write_slice_size(int slice_no, int size)
{
	slice_size_table[slice_no] = size;
	return;
}


//extern uint32_t slice_num_max;

// thread 12
//slice_max 4


int getStartSliceNumForThread(int thread_no)
{
	int index;
	if (slice_num_max < MAX_THREAD_NUM) {
		index = thread_no;
	} else 	if (MAX_THREAD_NUM != 1) {
		index = (slice_num_max / (MAX_THREAD_NUM - 1))	 * thread_no;
	} else {
		index = 0;
	}
	return index;
}
//thread 3
//slice_max  5
//slice 0 0
//slice 1 0
//slice 2 1
//slice 3 1
//slice 4 2


//thread 2
//slice_max  5
//slice 0 0
//slice 1 0
//slice 2 0
//slice 3 0
//slice 4 0

//thread 3
//slice max 1020
//slice 510 
int getThreadNumForSliceNo(int slice_no)
{
	int thread_no;
	int slice_num_for_thread;
	if (slice_num_max < MAX_THREAD_NUM) {
		thread_no = slice_no;
	} else 	if ((MAX_THREAD_NUM != 1) &&(MAX_THREAD_NUM != 2)) {
		slice_num_for_thread = slice_num_max / (MAX_THREAD_NUM -1);
		thread_no = slice_no / slice_num_for_thread;
	} else {
		thread_no = 0;
	}
	return thread_no;
}

int getSliceNumForThread(int thread_no) 
{
	int num;
	if (slice_num_max < MAX_THREAD_NUM) {
		if (slice_num_max  > thread_no) {
			num = 1;
		} else {
			num = 0;
		}
	} else if (MAX_THREAD_NUM != 1) {
		if ((MAX_THREAD_NUM -1) == thread_no) {
			num = slice_num_max % (MAX_THREAD_NUM -1);
		} else {
			num = slice_num_max / (MAX_THREAD_NUM -1) ;
		}
	} else {
			num = slice_num_max;
	}
	return num;
}

void *thread_start_routin(void *arg)
{
	struct thread_param *param =  (struct thread_param*)arg;


	int counter = 0;
	for(;;) {
#if 1
		//printf("start 1\n");
		pthread_mutex_lock(&slice_num_thread_mutex[param->thread_no]);
		while(slice_num_thread[param->thread_no] == 0) {
			counter = 0;
			pthread_cond_wait(&slice_num_thread_cond[param->thread_no], &slice_num_thread_mutex[param->thread_no]);
		}
		pthread_mutex_unlock(&slice_num_thread_mutex[param->thread_no]);
#endif
		//printf("start 2\n");
		int j;
		int index;
		int size=0;
//		int last_index =0;
		index = getStartSliceNumForThread(param->thread_no);
		for(j=0;j<slice_num_thread[param->thread_no];j++) {
			if (j==0) {
				initBitStream(slice_param[index].bitstream);
				//printf("bitstream %p %p %d \n", slice_param[index].bitstream, slice_param[index].bitstream->bitstream_buffer, index);
			}
//			last_index = index;
			uint16_t slice_size = encode_slice(&slice_param[index+j]);
			write_slice_size(slice_param[index+j].slice_no, slice_size);
//			index++;
			size += slice_size;
			//printf("size=%d\n", slice_size);
		}
//		int index = (counter * MAX_THREAD_NUM) + param->thread_no;
		//printf("wait_write_bitstream\n");
		wait_write_bitstream(param);
		//printf("start4 %p %d\n", slice_param[index].bitstream->bitstream_buffer, size*8);
		setByte(&write_bitstream, slice_param[index].bitstream->bitstream_buffer, size);
		if (slice_param[index+j-1].end == true) {
			//printf("start5 \n" );

			pthread_mutex_unlock(&end_frame_mutex);
		} else {
		    //printf("start6\n");
			start_write_next_bitstream(param);
		}
		//printf("start7\n");

		counter++;
		pthread_mutex_lock(&slice_num_thread_mutex[param->thread_no]);
		slice_num_thread[param->thread_no]=0;
		pthread_mutex_unlock(&slice_num_thread_mutex[param->thread_no]);
		//printf("start8\n");

	}
	
}



void start_write_bitstream(void) {
		pthread_mutex_unlock(&params[0].write_bitstream_my_mutex);
}

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



void encode_slices(struct encoder_param * param)
{
    uint32_t mb_x;
    uint32_t mb_y;
    uint32_t mb_x_max;
    mb_x_max = (param->horizontal+ 15 ) >> 4;


    slice_num_max = GetSliceNum(param->horizontal, param->vertical, param->slice_size_in_mb);

    int32_t slice_mb_count = param->slice_size_in_mb;
    mb_x = 0;
    mb_y = 0;

    /* write dummy slice size table */
    int32_t i;
    uint32_t slice_size_table_offset = (getBitSize(&write_bitstream)) >> 3 ;
    for (i = 0; i < slice_num_max ; i++) {
        uint16_t slice_size = 0x0;
        setByte(&write_bitstream, (uint8_t*)&slice_size, 2);
    }
    slice_mb_count = param->slice_size_in_mb;
    mb_x = 0;
    mb_y = 0;
    for (i = 0; i < slice_num_max ; i++) {

        while ((mb_x_max - mb_x) < slice_mb_count)
            slice_mb_count >>=1;

		int thread_no = getThreadNumForSliceNo(i);
        slice_param[i].slice_no = i;
        slice_param[i].luma_matrix = param->luma_matrix;
        slice_param[i].chroma_matrix = param->chroma_matrix;
        slice_param[i].qscale = param->qscale_table[i];
        slice_param[i].slice_size_in_mb= param->slice_size_in_mb;
        slice_param[i].horizontal= param->horizontal;
        slice_param[i].vertical= param->vertical;
        slice_param[i].y_data= (uint16_t*)param->y_data;
        slice_param[i].cb_data= (uint16_t*)param->cb_data;
        slice_param[i].cr_data= (uint16_t*)param->cr_data;
        slice_param[i].mb_x = mb_x;
        slice_param[i].mb_y = mb_y;
        slice_param[i].format_444 = param->format_444;
		//printf("Threadno %d %d\n",i , thread_no);
	    slice_param[i].bitstream = &slice_bitstream[thread_no];
	    slice_param[i].bitstream->bitstream_buffer = slice_bistream_buffer[thread_no];
		if (i==510) {
			//printf("%p %d\n", slice_bistream_buffer[thread_no], thread_no);
		}
		slice_param[i].working_buffer = params[thread_no].y_slice;


	   if (i == (slice_num_max -1)) {
		   //printf("end %d\n", i);
			slice_param[i].end = true;
		} else {
		   //printf("no end %d\n", i);
			slice_param[i].end = false;
		}

        mb_x += slice_mb_count;
        if (mb_x == mb_x_max ) {
            slice_mb_count = param->slice_size_in_mb;
            mb_x = 0;
            mb_y++;
        }
		
    }
	//printf("start thread %d %d\n", __LINE__, slice_num_max);
#if 1
	int j;
	for(j=0;j<MAX_THREAD_NUM;j++) {
		pthread_mutex_lock(&slice_num_thread_mutex[j]);
		slice_num_thread[j] = getSliceNumForThread(j);
		//printf("num %d %d\n", slice_num_thread[j], j);
		pthread_cond_signal(&slice_num_thread_cond[j]);
		pthread_mutex_unlock(&slice_num_thread_mutex[j]);
	}
#endif
	struct timeval startTime, endTime;

	gettimeofday(&startTime,NULL);
//	printf("s %d.%d\n", (int)startTime.tv_sec, (int)startTime.tv_usec);
	start_write_bitstream();


	frame_end_wait();
	gettimeofday(&endTime,NULL);
	//printf("e %d.%d\n", (int)endTime.tv_sec, (int)endTime.tv_usec);
#ifdef TIME_SCALE
	printf("end %d\n", (int)(endTime.tv_sec - startTime.tv_sec));
#endif
    for (i = 0; i < slice_num_max ; i++) {
        setSliceTalbeFlush(slice_size_table[i], slice_size_table_offset + (i * 2));
    }

}


uint8_t *encode_frame(struct encoder_param* param, uint32_t *encode_frame_size)
{

	write_bitstream.bitstream_buffer = bitstream_buffer;
    initBitStream(&write_bitstream);

    uint32_t frame_size_offset = getBitSize(&write_bitstream) >> 3 ;
    uint32_t frame_size = SET_DATA32(0x0); 
    setByte(&write_bitstream, (uint8_t*)&frame_size,4);

    uint32_t frame_identifier = SET_DATA32(0x69637066); //icpf


    setByte(&write_bitstream, (uint8_t*)&frame_identifier,4);

    set_frame_header(param);
    uint32_t picture_size_offset = (getBitSize(&write_bitstream)) >> 3 ;

    set_picture_header(param);

    encode_slices(param);
    uint32_t picture_end = (getBitSize(&write_bitstream)) >>  3 ;

    uint32_t tmp  = picture_end - picture_size_offset;
    uint32_t picture_size = SET_DATA32(tmp);

    setByteInOffset(&write_bitstream, picture_size_offset_, (uint8_t*)&picture_size, 4);


    uint8_t *ptr = getBitStream(&write_bitstream, encode_frame_size);
    uint32_t frame_size_data = SET_DATA32(*encode_frame_size);
    setByteInOffset(&write_bitstream, frame_size_offset, (uint8_t*)&frame_size_data , 4);
    return ptr;
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
	for(i=0;i<MAX_THREAD_NUM;i++) {
		params[i].write_bitstream_next_mutex = &params[(i+1)%MAX_THREAD_NUM].write_bitstream_my_mutex;
	}

	for(i=0;i<MAX_THREAD_NUM;i++) {
		ret = pthread_mutex_init(&slice_num_thread_mutex[i],NULL);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return -1;
		}
		ret = pthread_cond_init(&slice_num_thread_cond[i],NULL);
		if (ret != 0) {
			printf("%d\n", __LINE__);
			return -1;
		}
		slice_num_thread[i] = 0;
	}


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

	frame_end_mutex_init();

	return 0;
}



void encoder_init(void)
{
	vlc_init();
#ifdef PRE_CALC_COS
	dct_init();
#endif
	int ret = encoder_thread_init();
	if (ret != 0) {
		printf("%d", __LINE__);
		return;
	}

}




