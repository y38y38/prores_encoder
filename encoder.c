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

