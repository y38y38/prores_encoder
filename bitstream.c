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

#include "bitstream.h"



static uint8_t *tmp_buf = NULL;
uint32_t tmp_buf_byte_offset = 0;
uint32_t tmp_buf_bit_offset = 0;

uint8_t *initBuf(void)
{
    uint8_t *buf = (uint8_t*)malloc(1000*1000*1000);
    if (buf ==NULL) {
        printf("%s %d\n", __FUNCTION__, __LINE__);
        return NULL;
    }
    return buf;

}
void setBit(uint32_t buf, uint32_t size_of_bit)
{
    if (size_of_bit >= 24 )  {
        printf("error %s %d %d\n", __FUNCTION__, __LINE__, size_of_bit);
        return;
    }
    //printf("offset %x %d %x %d\n",tmp_buf_byte_offset,tmp_buf_bit_offset, buf, size_of_bit);
    
    uint32_t tmp = buf;
    uint32_t a = tmp_buf_bit_offset + size_of_bit;
    if (a > 32) {
        printf("error %s %d\n", __FUNCTION__, __LINE__);
        return ;
    }
    uint32_t b = 32 - a;
    tmp = tmp << b;
    uint8_t tmp_bit = *(tmp_buf + tmp_buf_byte_offset);
    tmp_bit = tmp_bit | ((uint8_t)(tmp>>24));
    *(tmp_buf + tmp_buf_byte_offset) = tmp_bit;
    //printf("set %x %x %x %x\n", tmp_bit,((uint8_t)(tmp>>16)),((uint8_t)(tmp>>8)),((uint8_t)(tmp)));
    *(tmp_buf + tmp_buf_byte_offset + 1) =  ((uint8_t)(tmp>>16));
    *(tmp_buf + tmp_buf_byte_offset + 2) =  ((uint8_t)(tmp>>8));
    *(tmp_buf + tmp_buf_byte_offset + 3) =  ((uint8_t)(tmp));
    //printf("bit %x %x\n", tmp_buf_byte_offset, tmp_bit);

    tmp_buf_byte_offset += (tmp_buf_bit_offset + size_of_bit) / 8;
    tmp_buf_bit_offset = (tmp_buf_bit_offset + size_of_bit) % 8;

}

void setByteInOffset(uint32_t offset, uint8_t *buf, uint32_t size_of_byte)
{
    
    if (tmp_buf_bit_offset != 0) {
        printf("error %s %d %d\n", __FUNCTION__, __LINE__,tmp_buf_bit_offset );
        return;
    }
    if (tmp_buf == NULL) {
        tmp_buf = initBuf();
        if (tmp_buf == NULL) {
            printf("%s %d\n", __FUNCTION__, __LINE__);
            return;
        }
    }
    memcpy(tmp_buf + offset, buf, size_of_byte);
}

void setByte(uint8_t *buf, uint32_t size_of_byte)
{
    
    if (tmp_buf_bit_offset != 0) {
        printf("error %s %d %d\n", __FUNCTION__, __LINE__,tmp_buf_bit_offset);
        return;
    }
    if (tmp_buf == NULL) {
        tmp_buf = initBuf();
        if (tmp_buf == NULL) {
            printf("%s %d\n", __FUNCTION__, __LINE__);
            return;
        }
    }
    memcpy(tmp_buf + tmp_buf_byte_offset , buf, size_of_byte);
    tmp_buf_byte_offset +=size_of_byte;
}
uint32_t getBitSize(void)
{
    return ((tmp_buf_byte_offset * 8) + tmp_buf_bit_offset );
}
uint8_t *getBitStream(uint32_t *size)
{
    if (tmp_buf_bit_offset != 0) {
        *size =  tmp_buf_byte_offset + ((tmp_buf_bit_offset + 7) / 8);
    } else {
        *size =  tmp_buf_byte_offset;
    }
    return tmp_buf;
}



