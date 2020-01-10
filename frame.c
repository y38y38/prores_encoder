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


#include "slice_table.h"
#include "dct.h"
#include "bitstream.h"
#include "encoder.h"
#include "slice.h"

uint16_t new_slice_table[15*68];

uint16_t getSliceSize(uint8_t mb_x, uint8_t mb_y)
{
    int32_t i;
    for(i=0;;i++) {
        if (slice_tables[i].x == 0xff) {
            break;
        } else if (mb_x == slice_tables[i].x) {
            if (mb_y == slice_tables[i].y) {
                return slice_tables[i].slice_size;
            }
        } 
    }
    printf("out of talbe %d %d\n", mb_x, mb_y);
    return 0xffff;
}
void setSliceTalbe(uint8_t mb_x, uint8_t mb_y) {
    uint16_t size = getSliceSize(mb_x,mb_y);
    uint16_t slice_size = SET_DATA16(size);
    if (slice_size== 0xFFFF ) {
        printf("%s %d\n", __FUNCTION__, __LINE__);
        return;
    }
    setByte((uint8_t*)&slice_size, 2);
    

}
void setSliceTalbeFlush(uint16_t size, uint32_t offset) {
    uint16_t slice_size = SET_DATA16(size);
    setByteInOffset(offset, (uint8_t*)&slice_size, 2);
    

}
/* get data for one slice */
uint16_t *getY(uint16_t *data, uint32_t mb_x, uint32_t mb_y, int32_t mb_size, int32_t horizontal, int32_t vertical)
{
    uint16_t *y = (uint16_t*)malloc(mb_size * 16 *16 * 2);
    if (y == NULL ) {
        printf("%d err\n", __LINE__);
        return NULL;
    }

    for(int32_t i = 0;i<16;i++) {
        memcpy(y + i * (mb_size * 16), data + (mb_x * 16) + (mb_y * 16) * horizontal+ i * horizontal, mb_size * 16 * 2);
    }
    return y;

}
/* get data for one slice */
uint16_t *getC(uint16_t *data, uint32_t mb_x, uint32_t mb_y, int32_t mb_size, int32_t horizontal, int32_t vertical)
{
    uint16_t *c = (uint16_t*)malloc(mb_size * 16 *8 * 2);
    if (c == NULL ) {
        printf("%d err\n", __LINE__);
        return NULL;
    }

    for(int32_t i = 0;i<16;i++) {
        memcpy(c + i * (mb_size * 8), data + (mb_x * 8) + (mb_y * 16) * (mb_size * 8)+ i * (horizontal/2), mb_size * 8 *2);
    }
    return c;

}
void encode_slices(struct encoder_param * param)
{
    uint32_t mb_x;
    uint32_t mb_y;
    uint32_t mb_x_max;
    uint32_t mb_y_max;
    mb_x_max = (param->horizontal+ 15 ) / 16;
    mb_y_max = (param->vertical+ 15) / 16;
    uint32_t slice_num_max;

    int32_t j = 0;

    uint32_t sliceSize = param->block_num; 
    uint32_t numMbsRemainingInRow = mb_x_max;
    uint32_t number_of_slices_per_mb_row_;

    do {
        while (numMbsRemainingInRow >= sliceSize) {
            j++;
            numMbsRemainingInRow  -=sliceSize;

        }
        sliceSize /= 2;
    } while(numMbsRemainingInRow  > 0);

    number_of_slices_per_mb_row_ = j;

    slice_num_max = number_of_slices_per_mb_row_ * mb_y_max;


    int32_t slice_mb_count = param->block_num;
    mb_x = 0;
    mb_y = 0;

    int32_t i;
    uint32_t slice_size_table_offset = (getBitSize()) /8 ;
    for (i = 0; i < slice_num_max ; i++) {

        while ((mb_x_max - mb_x) < slice_mb_count)
            slice_mb_count >>=1;
        //uint32_t table_offset = getBitSize();
        //printf("f1  %d\n", table_offset/8);
        setSliceTalbe(mb_x,mb_y);

        mb_x += slice_mb_count;
        if (mb_x == mb_x_max ) {
            slice_mb_count = param->block_num ;
            mb_x = 0;
            mb_y++;
        }


    }
    slice_mb_count = param->block_num;
    mb_x = 0;
    mb_y = 0;
//extern int32_t g_first;
    //g_first = 0;
    for (i = 0; i < slice_num_max ; i++) {

        while ((mb_x_max - mb_x) < slice_mb_count)
            slice_mb_count >>=1;

       //printf("%d %d\n", mb_x, mb_y);
       uint32_t size;
       uint16_t *y  = getY(param->y_data, mb_x,mb_y,slice_mb_count, param->horizontal, param->vertical);
       uint16_t *cb = getC(param->cb_data, mb_x,mb_y,slice_mb_count, param->horizontal, param->vertical);
       uint16_t *cr = getC(param->cr_data, mb_x,mb_y,slice_mb_count, param->horizontal, param->vertical);
       //size = encode_slice(y_data, cb_data, cr_data, mb_x, mb_y, slice_size);
       size = encode_slice(y, cb, cr, 0, 0);
       new_slice_table[i] = size;
       //printf("size = %d\n",size);

        mb_x += slice_mb_count;
        if (mb_x == mb_x_max ) {
            slice_mb_count = param->block_num;
            mb_x = 0;
            mb_y++;
                
        }
        //for debug
        //break;


    }

    slice_mb_count = param->block_num;
    mb_x = 0;
    mb_y = 0;
    for (i = 0; i < slice_num_max ; i++) {

        while ((mb_x_max - mb_x) < slice_mb_count)
            slice_mb_count >>=1;

        setSliceTalbeFlush(new_slice_table[i], slice_size_table_offset + (i * 2));
        //printf("f table offset %d \n", slice_size_table_offset + (i * 2));

        mb_x += slice_mb_count;
        if (mb_x == mb_x_max ) {
            slice_mb_count = param->block_num;
            mb_x = 0;
            mb_y++;
        }
        //for debug
        //break;


    }
    
}

uint32_t picture_size_offset_ = 0;
void set_picture_header(void)
{

    uint8_t picture_header_size = 0x8;//ToDo
    setBit(picture_header_size, 5);

    uint8_t reserved = 0x0;
    setBit(reserved , 3);

    picture_size_offset_ = (getBitSize()) /8 ;

    uint32_t picture_size = SET_DATA32(191616);//ToDo
    setByte((uint8_t*)&picture_size, 4);

    //uint16_t deprecated_number_of_slices =  SET_DATA16(771);//ToDo
    uint16_t deprecated_number_of_slices =  SET_DATA16(1020);//ToDo
    setByte((uint8_t*)&deprecated_number_of_slices , 0x2);


    uint8_t reserved2 = 0x0;
    setBit(reserved2 , 2);

    uint8_t log2_desired_slice_size_in_mb = 0x3;
    setBit(log2_desired_slice_size_in_mb, 2);

    uint8_t reserved3 = 0x0;
    setBit(reserved3 , 4);


}
void set_frame_header(struct encoder_param* param)
{
    uint16_t frame_header_size = SET_DATA16(0x94);//ToDo
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


    uint8_t chroma_format = 0x2;
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

    setByte(luma_matrix_, 64);
    setByte(chroma_matrix_, 64);


}

uint8_t *encode_frame(struct encoder_param* param, uint32_t *encode_frame_size)
{


    uint32_t frame_size = SET_DATA32(0x2ed1c); //ToDo
    setByte((uint8_t*)&frame_size,4);

    uint32_t frame_identifier = SET_DATA32(0x69637066);


    setByte((uint8_t*)&frame_identifier,4);

    set_frame_header(param);

    uint32_t picture_size_offset = (getBitSize()) /8 ;
    set_picture_header();

    encode_slices(param);
    uint32_t picture_end = (getBitSize()) /8 ;

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
    setByteInOffset(0, (uint8_t*)&frame_size_data , 4);
    return ptr;
}
void encoder_init(void)
{
    int32_t i,j;
    for(i=0;i<64;i++) {
        for(j=0;j<64;j++) {
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

