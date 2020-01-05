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


#include "qscale.h"
#include "slice_table.h"
#include "code_size.h"
#include "dct.h"
#include "bitstream.h"
#include "encoder.h"
#include "slice.h"

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
void set_frame_header(void)
{
    uint16_t frame_header_size = SET_DATA16(0x94);//ToDo
    setByte((uint8_t*)&frame_header_size, 0x2);

    uint8_t reserved = 0x0;
    setByte(&reserved, 0x1);

    uint8_t bitstream_version = 0x0;
    setByte(&bitstream_version, 0x1);


    uint32_t encoder_identifier = SET_DATA32(0x4c617663);
    setByte((uint8_t*)&encoder_identifier, 0x4);

    uint16_t horizontal_size = SET_DATA16(HORIZONTAL);
    setByte((uint8_t*)&horizontal_size , 0x2);

    uint16_t vertical_size = SET_DATA16(VIRTICAL);
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

uint8_t *encode_frame(uint16_t *y_data, uint16_t *cb_data, uint16_t *cr_data, uint32_t *encode_frame_size)
{


    uint32_t frame_size = SET_DATA32(0x2ed1c); //ToDo
    setByte((uint8_t*)&frame_size,4);

    uint32_t frame_identifier = SET_DATA32(0x69637066);


    setByte((uint8_t*)&frame_identifier,4);

    set_frame_header();

    uint32_t picture_size_offset = (getBitSize()) /8 ;
    set_picture_header();

    encode_slices(y_data, cb_data, cr_data);
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

