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
#include <stdbool.h>

#include "config.h"
#include "bitstream.h"





void setBit(struct bitstream *write_bitstream, uint32_t buf, uint32_t size_of_bit)
{
	if (write_bitstream->tmp_buf_byte_offset > (MAX_SLICE_BITSTREAM_SIZE - 4)) {
		printf("bit overflow\n");
	}
    if (size_of_bit >= 24 )  {
        printf("error %s %d %d\n", __FUNCTION__, __LINE__, size_of_bit);
        return;
    }
    
    uint32_t tmp = buf;
    uint32_t a = write_bitstream->tmp_buf_bit_offset + size_of_bit;
    if (a > 32) {
        printf("error %s %d\n", __FUNCTION__, __LINE__);
        return ;
    }
    uint32_t b = 32 - a;
    tmp = tmp << b;

	uint8_t tmp_bit;
	if (write_bitstream->tmp_buf_bit_offset != 0x0) {
    	tmp_bit = *(write_bitstream->tmp_buf + write_bitstream->tmp_buf_byte_offset);
	} else {
    	tmp_bit = 0;
	}
    tmp_bit = tmp_bit | ((uint8_t)(tmp>>24));
    *(write_bitstream->tmp_buf + write_bitstream->tmp_buf_byte_offset) = tmp_bit;
    //printf("set %x %x %x %x\n", tmp_bit,((uint8_t)(tmp>>16)),((uint8_t)(tmp>>8)),((uint8_t)(tmp)));
    *(write_bitstream->tmp_buf + write_bitstream->tmp_buf_byte_offset + 1) =  ((uint8_t)(tmp>>16));
    *(write_bitstream->tmp_buf + write_bitstream->tmp_buf_byte_offset + 2) =  ((uint8_t)(tmp>>8));
    *(write_bitstream->tmp_buf + write_bitstream->tmp_buf_byte_offset + 3) =  ((uint8_t)(tmp));
    //printf("bit %x %x\n", tmp_buf_byte_offset, tmp_bit);

    write_bitstream->tmp_buf_byte_offset += (write_bitstream->tmp_buf_bit_offset + size_of_bit) >> 3;
    write_bitstream->tmp_buf_bit_offset = (write_bitstream->tmp_buf_bit_offset + size_of_bit) & 7;
	//printf("offset %d %d\n", write_bitstream->tmp_buf_byte_offset, write_bitstream->tmp_buf_bit_offset);
}

void setByteInOffset(struct bitstream *write_bitstream, uint32_t offset, uint8_t *buf, uint32_t size_of_byte)
{
    
    if (write_bitstream->tmp_buf_bit_offset != 0) {
        printf("error %s %d %d\n", __FUNCTION__, __LINE__,write_bitstream->tmp_buf_bit_offset );
        return;
    }
    memcpy(write_bitstream->tmp_buf + offset, buf, size_of_byte);
}

void setByte(struct bitstream *write_bitstream, uint8_t *buf, uint32_t size_of_byte)
{
    
    if (write_bitstream->tmp_buf_bit_offset != 0) {
        printf("error %s %d %d\n", __FUNCTION__, __LINE__,write_bitstream->tmp_buf_bit_offset);
        return;
    }
    memcpy(write_bitstream->tmp_buf + write_bitstream->tmp_buf_byte_offset , buf, size_of_byte);
    write_bitstream->tmp_buf_byte_offset +=size_of_byte;
}

uint32_t getBitSize(struct bitstream *write_bitstream)
{
    return ((write_bitstream->tmp_buf_byte_offset * 8) + write_bitstream->tmp_buf_bit_offset );
}
uint8_t *getBitStream(struct bitstream *write_bitstream, uint32_t *size)
{
    if (write_bitstream->tmp_buf_bit_offset != 0) {
        *size =  write_bitstream->tmp_buf_byte_offset + ((write_bitstream->tmp_buf_bit_offset + 7) >> 3);
    } else {
        *size =  write_bitstream->tmp_buf_byte_offset;
    }
    return write_bitstream->tmp_buf;
}

void initBitStream(struct bitstream *write_bitstream)
{
    uint8_t *buf = (uint8_t*)write_bitstream->bitstream_buffer;
    write_bitstream->tmp_buf_byte_offset = 0;
    write_bitstream->tmp_buf_bit_offset = 0;
    write_bitstream->tmp_buf  = buf;

    return ;

}

