/**
 *
 * Copyright (c) 2020 Yuusuke Miyazaki
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 *
 **/

#ifndef __SLICE_H__
#define __SLICE_H__
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

#include "prores.h"
#include "encoder.h"


#define MAX_SLICE_DATA	(2048)

struct thread_param {
	int thread_no;
	pthread_mutex_t  write_bitstream_my_mutex;
	pthread_mutex_t  *write_bitstream_next_mutex;
	int16_t y_slice[MAX_SLICE_DATA];
	int16_t cb_slice[MAX_SLICE_DATA];
	int16_t cr_slice[MAX_SLICE_DATA];

};


struct Slice {
	int slice_no;
	uint32_t thread_num;
    uint8_t *luma_matrix;
    uint8_t *chroma_matrix;
    uint8_t qscale;
    uint32_t slice_size_in_mb;
    uint32_t horizontal;
    uint32_t vertical;
    uint16_t *y_data;
    uint16_t *cb_data;
    uint16_t *cr_data;
    uint32_t mb_x;
    uint32_t mb_y;
    bool format_444;
	bool end;
	struct bitstream *bitstream;
	struct bitstream *real_bitsteam;
	struct thread_param *thread_param;
};

uint32_t encode_slice(struct Slice *param);

uint8_t luma_matrix_[MATRIX_NUM];
uint8_t chroma_matrix_[MATRIX_NUM];
#endif
