
/**
 *
 * Copyright (c) 2020 Yuusuke Miyazaki
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 *
 **/

#ifndef __BITSTREAM_CUDA_H__
#define __BITSTREAM_CUDA_H__

#include "bitstream.h"

void initBitStream_cuda(struct bitstream *write_bitstream);

void setBit_cuda(struct bitstream *write_bitstream, uint32_t buf, uint32_t size_of_bit);

void setByteInOffset_cuda(struct bitstream *write_bitstream, uint32_t offset, uint8_t *buf, uint32_t size_of_byte);

void setByte_cuda(struct bitstream *write_bitstream, uint8_t *buf, uint32_t size_of_byte);

uint32_t getBitSize_cuda(struct bitstream *write_bitstream);

uint8_t *getBitStream_cuda(struct bitstream *write_bitstream, uint32_t *size);
#endif
