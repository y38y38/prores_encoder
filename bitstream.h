
/**
 *
 * Copyright (c) 2020 Yuusuke Miyazaki
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 *
 **/

#ifndef __BITSTREAM_H__
#define __BITSTREAM_H__

#define  SET_DATA32(x)    ((x & 0x000000ff) << 24 |  \
                        (x & 0x0000ff00) <<  8 | \
                        (x & 0x00ff0000) >>  8 | \
                        (x & 0xff000000) >> 24 ) 


#define  SET_DATA16(x)  ((x & 0x00ff) <<  8 | \
                         (x & 0xff00) >>  8 )


//#define MAX_SLICE_BITSTREAM_SIZE	(1024*1024) //1K
#define MAX_SLICE_BITSTREAM_SIZE	(1024*1024*10) //1M
//static uint8_t bitstream_buffer[MAX_BITSTREAM_SIZE];


struct bitstream {
	uint8_t *tmp_buf;
	uint32_t tmp_buf_byte_offset;
	uint32_t tmp_buf_bit_offset;
	uint8_t *bitstream_buffer;
};


void initBitStream(struct bitstream *write_bitstream);

void setBit(struct bitstream *write_bitstream, uint32_t buf, uint32_t size_of_bit);
void setByteInOffset(struct bitstream *write_bitstream, uint32_t offset, uint8_t *buf, uint32_t size_of_byte);
void setByte(struct bitstream *write_bitstream, uint8_t *buf, uint32_t size_of_byte);

uint32_t getBitSize(struct bitstream *write_bitstream);
uint8_t *getBitStream(struct bitstream *write_bitstream, uint32_t *size);
#endif
