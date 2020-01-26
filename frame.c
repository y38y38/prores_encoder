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
#include <math.h>
#include <stdbool.h>


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

/* get data for one slice */
uint16_t *getY(uint16_t *data, uint32_t mb_x, uint32_t mb_y, int32_t mb_size, int32_t horizontal, int32_t vertical)
{
    uint16_t *y = (uint16_t*)malloc(mb_size * MACRO_BLOCK_Y_HORIZONTAL * MACRO_BLOCK_Y_VERTICAL * sizeof(uint16_t));
    if (y == NULL ) {
        printf("%d err\n", __LINE__);
        return NULL;
    }

    for(int32_t i = 0;i<MACRO_BLOCK_Y_VERTICAL;i++) {
        memcpy(y + i * (mb_size * MACRO_BLOCK_Y_HORIZONTAL), 
               data + (mb_x * MACRO_BLOCK_Y_HORIZONTAL) + ((mb_y * MACRO_BLOCK_Y_VERTICAL) * horizontal) + (i * horizontal), 
               mb_size * MACRO_BLOCK_Y_HORIZONTAL * sizeof(uint16_t));
    }
    return y;

}
/* get data for one slice */
/* for 422 */
uint16_t *getC(uint16_t *data, uint32_t mb_x, uint32_t mb_y, int32_t mb_size, int32_t horizontal, int32_t vertical)
{
    uint16_t *c = (uint16_t*)malloc(mb_size * MACRO_BLOCK_422_C_HORIZONTAL * MACRO_BLOCK_422_C_VERTICAL * sizeof(uint16_t));
    if (c == NULL ) {
        printf("%d err\n", __LINE__);
        return NULL;
    }

    for(int32_t i = 0;i<MACRO_BLOCK_422_C_VERTICAL;i++) {
        memcpy(c + i * (mb_size * MACRO_BLOCK_422_C_HORIZONTAL), 
               data + (mb_x * MACRO_BLOCK_422_C_HORIZONTAL) + ((mb_y * MACRO_BLOCK_422_C_VERTICAL) * (horizontal/2)) + (i * (horizontal/2)), 
               mb_size * MACRO_BLOCK_422_C_HORIZONTAL * sizeof(uint16_t));
    }
    return c;

}
void encode_slices(struct encoder_param * param)
{
    uint32_t mb_x;
    uint32_t mb_y;
    uint32_t mb_x_max;
    mb_x_max = (param->horizontal+ 15 ) / 16;
    uint32_t slice_num_max;


    slice_num_max = GetSliceNum(param->horizontal, param->vertical, param->slice_size_in_mb);

    int32_t slice_mb_count = param->slice_size_in_mb;
    mb_x = 0;
    mb_y = 0;

    /* write dummy slice size table */
    int32_t i;
    uint32_t slice_size_table_offset = (getBitSize()) / 8 ;
    for (i = 0; i < slice_num_max ; i++) {
        uint16_t slice_size = 0x0;
        setByte((uint8_t*)&slice_size, 2);
    }

    uint16_t *slice_size_table = (uint16_t*)malloc(slice_num_max * sizeof(uint16_t));
    if (slice_size_table  == NULL) {
        printf("err %d\n", __LINE__);
        return; 
    }
    slice_mb_count = param->slice_size_in_mb;
    mb_x = 0;
    mb_y = 0;
    for (i = 0; i < slice_num_max ; i++) {

        while ((mb_x_max - mb_x) < slice_mb_count)
            slice_mb_count >>=1;

       //printf("%d %d\n", mb_x, mb_y);
       uint32_t size;
       uint16_t *y  = getY(param->y_data, mb_x,mb_y,slice_mb_count, param->horizontal, param->vertical);
        uint16_t *cb;
        uint16_t *cr;
        if (param->format_444 ==  true) {
            cb = getY(param->cb_data, mb_x,mb_y,slice_mb_count, param->horizontal, param->vertical);
            cr = getY(param->cr_data, mb_x,mb_y,slice_mb_count, param->horizontal, param->vertical);
       } else {
            cb = getC(param->cb_data, mb_x,mb_y,slice_mb_count, param->horizontal, param->vertical);
            cr = getC(param->cr_data, mb_x,mb_y,slice_mb_count, param->horizontal, param->vertical);
       }

#if 0
#if 1
       for(int i = 0;i<slice_mb_count*16*16;i++) {
           //*(y + i) = 0x200;
       }
       if (mb_y %2) {
       for(int i = 0;i<slice_mb_count*16*16/2;i++) {
           *(cb + i) = 0x200;
       }
       for(int i = 0;i<slice_mb_count*16*16/2;i++) {
           *(cr + i) = 0x200;
       }
       }
#else
       memset(y, 0x00, slice_mb_count*16*16*2);

       memset(cb, 0x00, slice_mb_count*16*16*2/2);
       memset(cr, 0x00, slice_mb_count*16*16*2/2);
#endif
#endif

       struct Slice slice_param;
       slice_param.luma_matrix = param->luma_matrix;
       slice_param.chroma_matrix = param->chroma_matrix;
       slice_param.qscale = param->qscale_table[i];
       slice_param.slice_size_in_mb= param->slice_size_in_mb;
       slice_param.horizontal= param->horizontal;
       slice_param.vertical= param->vertical;
       slice_param.y_data= y;
       slice_param.cb_data= cb;
       slice_param.cr_data= cr;
       slice_param.mb_x = 0;
       slice_param.mb_y = 0;
       slice_param.format_444 = param->format_444;

       //size = encode_slice(y_data, cb_data, cr_data, mb_x, mb_y, slice_size);
       /* need mb_x = 0 and mb_y = 0 becase getY and getC takas data to mb_x=0 and mb_y=0 position . */
       size = encode_slice(&slice_param);
       slice_size_table[i] = size;
       //printf("size = %d\n",size);

        mb_x += slice_mb_count;
        if (mb_x == mb_x_max ) {
            slice_mb_count = param->slice_size_in_mb;
            mb_x = 0;
            mb_y++;
        }

    }

    for (i = 0; i < slice_num_max ; i++) {
        setSliceTalbeFlush(slice_size_table[i], slice_size_table_offset + (i * 2));
    }
    free(slice_size_table);
    
}

uint32_t picture_size_offset_ = 0;
void set_picture_header(struct encoder_param* param)
{

    uint8_t picture_header_size = 0x8;
    setBit(picture_header_size, 5);

    uint8_t reserved = 0x0;
    setBit(reserved , 3);

    picture_size_offset_ = (getBitSize()) /8 ;

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

    uint32_t frame_size_offset = getBitSize() / 8 ;
    uint32_t frame_size = SET_DATA32(0x0); 
    setByte((uint8_t*)&frame_size,4);

    uint32_t frame_identifier = SET_DATA32(0x69637066); //icpf


    setByte((uint8_t*)&frame_identifier,4);

    set_frame_header(param);

    uint32_t picture_size_offset = (getBitSize()) / 8 ;
    set_picture_header(param);

    encode_slices(param);
    uint32_t picture_end = (getBitSize()) / 8 ;

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
}
int32_t GetSliceNum(int32_t horizontal, int32_t vertical, int32_t sliceSize)
{
    int32_t mb_x_max = (horizontal + 15)  / 16;
    int32_t mb_y_max = (vertical + 15) / 16;

    int32_t slice_num_max;

    int32_t numMbsRemainingInRow = mb_x_max;
    int32_t number_of_slices_per_mb_row;
    int j = 0;

    do {
        while (numMbsRemainingInRow >= sliceSize) {
            j++;
            numMbsRemainingInRow  -=sliceSize;

        }
        sliceSize /= 2;
    } while(numMbsRemainingInRow  > 0);

    number_of_slices_per_mb_row = j;

    slice_num_max = number_of_slices_per_mb_row * mb_y_max;
    return slice_num_max;

}
uint32_t GetEncodeHorizontal(int32_t horizontal)
{
    return ((horizontal + 15)  / 16) * 16;
}
uint32_t GetEncodeVertical(int32_t vertical)
{
    return ((vertical + 15)  / 16) * 16;
}

