
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

struct bitstream {
	uint8_t *tmp_buf;
	uint32_t tmp_buf_byte_offset;
	uint32_t tmp_buf_bit_offset;
	uint8_t *bitstream_buffer;
};


extern void initBitStream(void);
extern void setBit(uint32_t buf, uint32_t size_of_bit);
extern void setByteInOffset(uint32_t offset, uint8_t *buf, uint32_t size_of_byte);

extern void setByte(uint8_t *buf, uint32_t size_of_byte);

extern uint32_t getBitSize(void);

extern uint8_t *getBitStream(uint32_t *size);
#endif
