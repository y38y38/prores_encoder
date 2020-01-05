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


uint8_t luma_matrix_[64] =
{
0x04, 0x04, 0x05, 0x05, 0x06, 0x07, 0x07, 0x09, 
0x04, 0x04, 0x05, 0x06, 0x07, 0x07, 0x09, 0x09, 
0x05, 0x05, 0x06, 0x07, 0x07, 0x09, 0x09, 0x0a, 
0x05, 0x05, 0x06, 0x07, 0x07, 0x09, 0x09, 0x0a, 
0x05, 0x06, 0x07, 0x07, 0x08, 0x09, 0x0a, 0x0c, 
0x06, 0x07, 0x07, 0x08, 0x09, 0x0a, 0x0c, 0x0f, 
0x06, 0x07, 0x07, 0x09, 0x0a, 0x0b, 0x0e, 0x11, 
0x07, 0x07, 0x09, 0x0a, 0x0b, 0x0e, 0x11, 0x15, 

};

uint8_t chroma_matrix_[64] =
{
0x04, 0x04, 0x05, 0x05, 0x06, 0x07, 0x07, 0x09, 
0x04, 0x04, 0x05, 0x06, 0x07, 0x07, 0x09, 0x09, 
0x05, 0x05, 0x06, 0x07, 0x07, 0x09, 0x09, 0x0a, 
0x05, 0x05, 0x06, 0x07, 0x07, 0x09, 0x09, 0x0a, 
0x05, 0x06, 0x07, 0x07, 0x08, 0x09, 0x0a, 0x0c, 
0x06, 0x07, 0x07, 0x08, 0x09, 0x0a, 0x0c, 0x0f, 
0x06, 0x07, 0x07, 0x09, 0x0a, 0x0b, 0x0e, 0x11, 
0x07, 0x07, 0x09, 0x0a, 0x0b, 0x0e, 0x11, 0x15, 
};
#if 0
#define  SET_DATA32(x)    ((x & 0x000000ff) << 24 |  \
                        (x & 0x0000ff00) <<  8 | \
                        (x & 0x00ff0000) >>  8 | \
                        (x & 0xff000000) >> 24 ) 


#define  SET_DATA16(x)  ((x & 0x00ff) <<  8 | \
                         (x & 0xff00) >>  8 )
#endif
uint16_t new_slice_table[15*68];

uint8_t getQscale(uint8_t mb_x, uint8_t mb_y)
{
    return 3;
}
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
int32_t g_print  = 0;
void aprint_block(int16_t *block)
{

    int32_t x,y;
    for (y=0;y<8;y++) {
        for (x=0;x<8;x++) {
            printf("%d ", block[(y * 8) + x]);
        }
        printf("\n");
    }
    printf("\n");
}
void print_mb(int16_t *mb)
{
    int32_t i;
    for(i=0;i<4;i++) {
        aprint_block(mb + (i*64));
    }
        

}
void print_mb_cb(int16_t *mb)
{
    int32_t i;
    for(i=0;i<2;i++) {
        aprint_block(mb + (i*64));
    }
        

}
void print_slice_cb(int16_t *slice, int32_t mb_num)
{
    int32_t i;
    for(i=0;i<mb_num;i++) {
        print_mb_cb(slice + (i*64)*2);
    }
        
    if (g_print == 1) {
        //printf("\n");
    }

}
void print_slice(int16_t *slice, int32_t mb_num)
{
    int32_t i;
    for(i=0;i<mb_num;i++) {
        print_mb(slice + (i*64)*4);
    }
        
    if (g_print == 1) {
        //printf("\n");
    }

}
void print_pixels(int16_t *slice, int32_t mb_num)
{
    int32_t i;
    for(i=0;i<mb_num*4*64;i++) {
        printf("%d\n", slice[i]);
    }
        

}
uint16_t getYCodeSize(uint8_t mb_x, uint8_t mb_y)
{
    int32_t i;
    for(i=0;;i++) {
        if (code_sizes[i].x == 0xff) {
            break;
        } else if (mb_x == code_sizes[i].x) {
            if (mb_y == code_sizes[i].y) {
                return code_sizes[i].coded_size_of_y_data;
            }
        } 
    }
    printf("out of talbe %d %d\n", mb_x, mb_y);
    return 0xffff;
}
uint16_t getCbCodeSize(uint8_t mb_x, uint8_t mb_y)
{
    int32_t i;
    for(i=0;;i++) {
        if (code_sizes[i].x == 0xff) {
            break;
        } else if (mb_x == code_sizes[i].x) {
            if (mb_y == code_sizes[i].y) {
                return code_sizes[i].coded_size_of_cb_data;
            }
        } 
    }
    printf("out of talbe %d %d\n", mb_x, mb_y);
    return 0xffff;
}
#if 0
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
#endif



int32_t abs22(int32_t val)
{
    if (val < 0) {
        //printf("m\n");
        return val * -1;
    } else {
        //printf("p\n");
        return val;
    }
}
void golomb_rice_code(int32_t k, uint32_t val)
{
    int32_t q  = floor( val / pow(2,k));
    if (k ==0) {
        //uint32_t tmp = pow(2,k);
        //uint32_t r = val % tmp;
        if (q != 0) {
            setBit(0,q);
            //printf("1: 0 %d\n", q);
        }
        setBit(1,1);
        //printf("1: 1 1    k:%d q:%d r:%d\n", k, q, r);
    } else {
        uint32_t tmp = pow(2,k);
        uint32_t r = val % tmp;
        uint32_t codeword = (1 << k) | r;
        setBit(codeword, q + 1 + k );
        //printf("2: %x %d\n",codeword, q+1+k);
    }
    return;
}
void exp_golomb_code(int32_t k, uint32_t val)
{
    int32_t q = floor(log2(val + pow(2, k))) - k;

    //printf("pow %f %d ", pow(2,k),val);
    uint32_t sum = val + pow(2, k);
    int32_t codeword_length = (2 * q) + k + 1;

    setBit(sum, codeword_length);
    //printf("3: val:%x q:%d k:%d bits:%d k:%d\n", sum,q, k, codeword_length,k );
    return;
}
void rice_exp_combo_code(int32_t last_rice_q, int32_t k_rice, int32_t k_exp, uint32_t val)
{
    //printf("recc ");
    uint32_t value = (last_rice_q + 1) * pow(2, k_rice);
    if (val < value) {
        golomb_rice_code(k_rice, val);
    } else {
        setBit(0,last_rice_q + 1);
        //printf("0: val:0 bits:%d\n", last_rice_q + 1);
        exp_golomb_code(k_exp, val - value);
    }
    return;
}
void entropy_encode_dc_coefficient(bool first, int32_t abs_previousDCDiff , int val)
{
    if (first) {
        exp_golomb_code(5, val);
    } else if (abs_previousDCDiff == 0) {
        exp_golomb_code(0, val);
    } else if (abs_previousDCDiff == 1) {
        exp_golomb_code(1, val);
    } else if (abs_previousDCDiff == 2) {
        rice_exp_combo_code(1,2,3, val);
    } else {
        exp_golomb_code(3, val);
    }
    return;

}
void encode_vlc_codeword_ac_run(int32_t previousRun, int32_t val)
{
    if ((previousRun== 0)||(previousRun== 1)) {
        rice_exp_combo_code(2,0,1, val);
    } else if ((previousRun== 2)||(previousRun== 3)) {
        rice_exp_combo_code(1,0,1, val);
    } else if (previousRun== 4) {
        exp_golomb_code(0, val);
    } else if ((previousRun>= 5) && (previousRun <= 8))  {
        rice_exp_combo_code(1,1,2, val);
    } else if ((previousRun>= 9) && (previousRun <= 14))  {
        exp_golomb_code(1, val);
    } else {
        exp_golomb_code(2, val);
    }
    return;

}
void encode_vlc_codeword_ac_level(int32_t previousLevel, int32_t val)
{
    if (previousLevel== 0) {
        rice_exp_combo_code(2,0,2, val);
    } else if (previousLevel== 1) {
        rice_exp_combo_code(1,0,1, val);
    } else if (previousLevel== 2) {
        rice_exp_combo_code(2,0,1, val);
    } else if (previousLevel == 3)  {
        exp_golomb_code(0, val);
    } else if ((previousLevel>= 4) && (previousLevel<= 7))  {
        exp_golomb_code(1, val);
    } else {
        exp_golomb_code(2, val);
    }
    return;

}

int32_t Signedintegertosymbolmapping(int32_t val)
{
    uint32_t sn;
    if (val >=0 ) {
        sn = 2 * abs22(val);

    } else {
        sn = 2 * abs22(val) - 1;
    }
    return sn;
}
void entropy_encode_dc_coefficients(int16_t*coefficients, int32_t numBlocks)
{
    int32_t DcCoeff;
    int32_t val;
    int32_t previousDCCoeff;
    int32_t previousDCDiff;
    int32_t n;
    int32_t dc_coeff_difference;
    int32_t abs_previousDCDiff;

    DcCoeff = (coefficients[0]) ;
    val = Signedintegertosymbolmapping(DcCoeff);
    entropy_encode_dc_coefficient(true, 0, val);

    
    previousDCCoeff= DcCoeff;
    previousDCDiff = 3;
    n = 1;
    while( n <numBlocks) {
        DcCoeff = (coefficients[n++ * 64]); 
        dc_coeff_difference = DcCoeff - previousDCCoeff;
        if (previousDCDiff < 0) {
            dc_coeff_difference *= -1;
        }
        val = Signedintegertosymbolmapping(dc_coeff_difference);
        abs_previousDCDiff = abs22(previousDCDiff );
        entropy_encode_dc_coefficient(false, abs_previousDCDiff, val);
        previousDCDiff = DcCoeff - previousDCCoeff;
        previousDCCoeff= DcCoeff;

    }
    return ;
}

//from figure 4
const uint8_t block_pattern_scan_table[64] = {
     0,  1,  4,  5, 16, 17, 21, 22,
     2,  3,  6,  7, 18, 20, 23, 28,
     8,  9, 12, 13, 19, 24, 27, 29,
    10, 11, 14, 15, 25, 26, 30, 31,
    32, 33, 37, 38, 45, 46, 53, 54,
    34, 36, 39, 44, 47, 52, 55, 60,
    35, 40, 43, 48, 51, 56, 59, 61,
    41, 42, 49, 50, 57, 58, 62, 63,
};
uint8_t block_pattern_scan_read_order_table[64];


#define MAX_COEFFICIENT_NUM_PER_BLOCK (64)
uint32_t entropy_encode_ac_coefficients(int16_t*coefficients, int32_t numBlocks)
{
    int32_t block;
    int32_t conefficient;
    int32_t run;
    int32_t level;
    int32_t abs_level_minus_1;
    int32_t previousRun = 4;
    int32_t previousLevelSymbol = 1;
    int32_t position;

    run = 0;

    //start is 1 because 0 equal dc position
    for (conefficient = 1; conefficient< MAX_COEFFICIENT_NUM_PER_BLOCK; conefficient++) {
        position = block_pattern_scan_read_order_table[conefficient];
        for (block=0; block < numBlocks; block++) {
            level = coefficients[(block * MAX_COEFFICIENT_NUM_PER_BLOCK) + position] ;

            if (level != 0) {
                encode_vlc_codeword_ac_run(previousRun, run);

                abs_level_minus_1 = abs22(level) - 1;
                encode_vlc_codeword_ac_level( previousLevelSymbol, abs_level_minus_1);
                if (level >=0) {
                    setBit(0,1);
                } else {
                    setBit(1,1);
                }

                previousRun = run;
                previousLevelSymbol = abs_level_minus_1;
                run    = 0;

            } else {
                run++;
            }
        }
    }
    return 0;
}

uint16_t * getYDataToBlock(uint16_t*y_data,uint32_t mb_x, uint32_t mb_y, uint32_t mb_size)
{
    uint16_t pixel_data[8 * 4 * 64 * sizeof(uint16_t)];
    // macro block num * block num per macro  block * pixel num per block * pixel size
    uint16_t *y_slice = (uint16_t*)malloc((mb_size * 4) * 64 * sizeof(uint16_t));
    if (y_slice == NULL) {
        printf("errr  %d\n", __LINE__);
        return NULL;
    }
    int32_t i;
    for (i=0;i<16;i++) {
        memcpy(pixel_data + (i * (16 * mb_size)), y_data + (mb_x * 16)  + ((mb_y * 16) * mb_size*16) + (i * mb_size*16), 16 * mb_size * sizeof(uint16_t));
    }
    int32_t vertical;
    int32_t block;
    int32_t block_position = 0;
    for (i=0;i<mb_size;i++) {
        for (block = 0 ; block < 4;block++) {
            for(vertical= 0;vertical<8;vertical++) {
                if (block == 0) {
                    block_position = 0;
                } else if (block == 1) {
                    block_position =  8;
                } else if (block == 2) {
                    block_position =  mb_size * 16 * 8;
                } else {
                    block_position =  (mb_size * 16 * 8) + 8;
                }
                memcpy(y_slice + (i * (16 * 16)) +  (block * 64) + (vertical * 8) , 
                        pixel_data + ( 16 * i) + block_position + vertical * (mb_size * 16),
                        8 * sizeof(uint16_t));
            }
        }

    }
    return y_slice;



}
uint16_t * getCbDataToBlock(uint16_t*cb_data,uint32_t mb_x, uint32_t mb_y, uint32_t mb_size)
{
    //test_data(cb_data);
    int32_t i;
    //for (i=0;i<16;i++) {

    uint16_t pixel_data[8 * 4 * 64 * sizeof(uint16_t)];
    // macro block num * block num per macro  block * pixel num per block * pixel size
    uint16_t *cb_slice = (uint16_t*)malloc((mb_size * 4) * 64 * sizeof(uint16_t));
    if (cb_slice == NULL) {
        printf("errr  %d\n", __LINE__);
        return NULL;
    }
    for (i=0;i<16;i++) {
        memcpy(pixel_data + (i * (8 * mb_size)), cb_data + (mb_x * 8)  + ((mb_y * 16) * mb_size*16/2  ) + (i * (mb_size*16/2  )), 8 * mb_size * sizeof(uint16_t));
    }

    //memset(pixel_data, 0x0, 64);
    int32_t vertical;
    int32_t block;
    int32_t block_position = 0;
    //printf("aa %p\n", pixel_data);
    for (i=0;i<mb_size;i++) {
        for (block = 0 ; block < 2;block++) { //4:2:2
            for(vertical= 0;vertical<8;vertical++) {
                if (block == 0) {
                    block_position = 0;
                } else  {
                    block_position =  mb_size * 8 * 8;
                }
                memcpy(cb_slice + (i * (8 * 16)) +  (block * 64) + (vertical * 8) , 
                        pixel_data + ( 8 * i) + block_position + vertical * (mb_size * 8),
                        8 * sizeof(uint16_t));
                //printf("%x %d ", pixel_data[0], ( 8 * i) + block_position + vertical * (mb_size * 8));
            }
        }

    }
    return cb_slice;



}
void encode_qt(int16_t *y_slice, uint8_t *qmat, int32_t  block_num)
{

    int16_t *data;
    int32_t i,j;
    for (i = 0; i < block_num; i++) {
        data = y_slice + (i*64);
        for (j=0;j<64;j++) {
            data[j] = data [j]/ ( qmat[j]) ;
        }

    }
}
void encode_qscale(int16_t *y_slice, uint8_t scale, int32_t  block_num)
{

    int16_t *data = y_slice;
    int32_t i,j;
    for (i = 0; i < block_num; i++) {
        data = y_slice + (i*64);
        for (j=0;j<64;j++) {
            //if (j==0) {
                data[j] = data [j] /scale;
                //data[j] = data [j];
            //}
        }

    }
}
void pre_quant(int16_t *y_slice, int32_t  block_num)
{

    int16_t *data;
    int32_t i,j;
    for (i = 0; i < block_num; i++) {
        data = y_slice + (i*64);
        for (j=0;j<64;j++) {
            data[j] = data [j] * 8;
        }

    }
}
void pre_dct(int16_t *y_slice, int32_t  block_num)
{

    int16_t *data;
    int32_t i,j;
    for (i = 0; i < block_num; i++) {
        data = y_slice + (i*64);
        for (j=0;j<64;j++) {
            data[j] = (data[j] / 2) - 256;
        }

    }
}
uint32_t encode_slice_y(uint16_t*y_data, uint32_t mb_x, uint32_t mb_y, int32_t scale)
{
    //print_slice(y_data, 8);
    //printf("%s start\n", __FUNCTION__);
    uint32_t start_offset= getBitSize();
    //printf("start_offset %d\n", start_offset);

    int16_t *y_slice = getYDataToBlock(y_data,mb_x,mb_y,8);

    //printf("before\n");
    //print_pixels(y_slice, 8);

    //printf("pre\n");
    pre_dct(y_slice, 32);
    //printf("after\n");
    //print_pixels(y_slice, 8);

    int32_t i;
    //print_slice(y_slice, 8);
    for (i = 0;i< 4*8;i++) {
        dct_block(&y_slice[i* (8*8)]);
    }
    //print_slice(y_slice, 8);
    //print_slice(y_slice, 8);
    //

    pre_quant(y_slice, 32);

    encode_qt(y_slice, luma_matrix_, 32);
    //print_slice(y_slice, 8);
    encode_qscale(y_slice, scale , 32);


    entropy_encode_dc_coefficients(y_slice, 32);
    entropy_encode_ac_coefficients(y_slice, 32);

    //byte aliened
    uint32_t size  = getBitSize();
    if (size % 8 )  {
        setBit(0x0, 8 - (size % 8));
    }
    uint32_t current_offset = getBitSize();
    //printf("current_offset %d\n",current_offset );
    //printf("%s end\n", __FUNCTION__);
    return ((current_offset - start_offset)/8);
}
uint32_t encode_slice_cb(uint16_t*cb_data, uint32_t mb_x, uint32_t mb_y, int32_t scale)
{
    //printf("cb start\n");
    uint32_t start_offset= getBitSize();

    uint16_t *cb_slice = getCbDataToBlock(cb_data,mb_x,mb_y,8);

    pre_dct(cb_slice, 16);

    int32_t i;
    //memset(cb_slice, 0x0, 64);
    //extern int g_first;
    //g_first = 0;
    //print_slice_cb(cb_slice, 4);
    for (i = 0;i< 2*8;i++) {
        dct_block((int16_t*)&cb_slice[i* (8*8)]);
    }
    //printf("af\n");
    //print_slice_cb(cb_slice, 4);
    pre_quant(cb_slice, 16);

    encode_qt(cb_slice, chroma_matrix_, 16);
    encode_qscale(cb_slice,scale , 16);

    entropy_encode_dc_coefficients(cb_slice, 16);
    entropy_encode_ac_coefficients(cb_slice, 16);

    //byte aliened
    uint32_t size  = getBitSize();
    if (size % 8 )  {
        setBit(0x0, 8 - (size % 8));
    }
    uint32_t current_offset = getBitSize();
    //printf("%s end\n", __FUNCTION__);
    return ((current_offset - start_offset)/8);
}
uint32_t encode_slice_cr(uint16_t*cr_data, uint32_t mb_x, uint32_t mb_y, int32_t scale)
{
    //printf("%s start\n", __FUNCTION__);
    uint32_t start_offset= getBitSize();

    uint16_t *cr_slice = getCbDataToBlock(cr_data,mb_x,mb_y,8);

    pre_dct(cr_slice, 16);

    int32_t i;
    //extern int32_t g_first;
    //g_first = 0;
    //print_slice_cb(cr_slice, 4);
    for (i = 0;i< 4*8;i++) {
        dct_block((int16_t*)&cr_slice[i* (8*8)]);
    }
    //print_slice_cb(cr_slice, 4);
    pre_quant(cr_slice, 16);
    encode_qt(cr_slice, chroma_matrix_, 16);
    encode_qscale(cr_slice,scale , 16);

    entropy_encode_dc_coefficients(cr_slice, 16);
    entropy_encode_ac_coefficients(cr_slice, 16);
    //byte aliened
    uint32_t size  = getBitSize();
    if (size % 8 )  {
        setBit(0x0, 8 - (size % 8));
    }
    uint32_t current_offset = getBitSize();
    //printf("%s end\n", __FUNCTION__);
    return ((current_offset - start_offset)/8);
}

uint32_t encode_slice(uint16_t *y_data, uint16_t *cb_data, uint16_t *cr_data, uint32_t  mb_x, uint32_t mb_y, uint16_t slice_size)
{
    uint32_t start_offset= getBitSize();
    uint8_t qscale = getQscale(mb_x,mb_y);
    if (qscale == 0xFF ) {
        printf("%s %d\n", __FUNCTION__, __LINE__);
        return 0;
    }
    uint8_t slice_header_size = 6;

    setBit(slice_header_size , 5);

    uint8_t reserve =0x0;
    setBit(reserve, 3);

    setByte(&qscale, 1);

    uint32_t code_size_of_y_data_offset = getBitSize();
    code_size_of_y_data_offset = code_size_of_y_data_offset >> 3;
    uint16_t size = getYCodeSize(mb_x,mb_y);
    uint16_t coded_size_of_y_data = SET_DATA16(size);
    setByte((uint8_t*)&coded_size_of_y_data , 2);

    uint32_t code_size_of_cb_data_offset = getBitSize();
    code_size_of_cb_data_offset = code_size_of_cb_data_offset >> 3 ;
    size = getCbCodeSize(mb_x,mb_y);
    uint16_t coded_size_of_cb_data = SET_DATA16(size);
    setByte((uint8_t*)&coded_size_of_cb_data , 2);



    //printf("%s start\n", __FUNCTION__);

    size = (uint16_t)encode_slice_y(y_data, mb_x, mb_y, qscale);
    //exit(1);
    uint16_t y_size  = SET_DATA16(size);
    //printf("y %d %x\n", size, size);
    size = (uint16_t)encode_slice_cb(cb_data, mb_x, mb_y, qscale);
    uint16_t cb_size = SET_DATA16(size);
    //printf("cb %d\n", size);
    //exit(1);
#if 1
    size = (uint16_t)encode_slice_cr(cr_data, mb_x, mb_y, qscale);
    //uint16_t cr_size = SET_DATA16(size);
    //printf("cr%d\n", size);
#endif
    setByteInOffset(code_size_of_y_data_offset , (uint8_t *)&y_size, 2);
    //printf("%d %x\n",code_size_of_y_data_offset,code_size_of_y_data_offset); 
    setByteInOffset(code_size_of_cb_data_offset , (uint8_t *)&cb_size, 2);
    uint32_t current_offset = getBitSize();
    return ((current_offset - start_offset)/8);
}
//uint8_t encode_frame(
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
uint16_t *getY(uint16_t *data, uint32_t mb_x, uint32_t mb_y, int32_t mb_size)
{
    uint16_t *y = (uint16_t*)malloc(mb_size * 16 *16 * 2);
    if (y == NULL ) {
        printf("%d err\n", __LINE__);
        return NULL;
    }

    for(int32_t i = 0;i<16;i++) {
        memcpy(y + i * (mb_size * 16), data + (mb_x * 16) + (mb_y * 16) * HORIZONTAL + i * HORIZONTAL, mb_size * 16 * 2);
    }
    return y;

}
uint16_t *getC(uint16_t *data, uint32_t mb_x, uint32_t mb_y, int32_t mb_size)
{
    uint16_t *c = (uint16_t*)malloc(mb_size * 16 *8 * 2);
    if (c == NULL ) {
        printf("%d err\n", __LINE__);
        return NULL;
    }

    for(int32_t i = 0;i<16;i++) {
        memcpy(c + i * (mb_size * 8), data + (mb_x * 8) + (mb_y * 16) * (mb_size * 8)+ i * (HORIZONTAL/2), mb_size * 8 *2);
    }
    return c;

}
void encode_slices(uint16_t *y_data, uint16_t *cb_data, uint16_t *cr_data)
{
    uint32_t mb_x;
    uint32_t mb_y;
    uint32_t mb_x_max;
    uint32_t mb_y_max;
    mb_x_max = (HORIZONTAL+ 15 )>> 4;
    mb_y_max = VIRTICAL2/ 16;
    uint32_t slice_num_max;
    uint32_t log2_desired_slice_size_in_mb = 3;

    int32_t j = 0;

    uint32_t sliceSize = 1 << log2_desired_slice_size_in_mb; 
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


    int32_t slice_mb_count = 1 << log2_desired_slice_size_in_mb;
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
            slice_mb_count = 1 << log2_desired_slice_size_in_mb ;
            mb_x = 0;
            mb_y++;
        }


    }
    slice_mb_count = 1 << log2_desired_slice_size_in_mb;
    mb_x = 0;
    mb_y = 0;
//extern int32_t g_first;
    //g_first = 0;
    for (i = 0; i < slice_num_max ; i++) {

        while ((mb_x_max - mb_x) < slice_mb_count)
            slice_mb_count >>=1;

       uint16_t slice_size = getSliceSize(mb_x,mb_y);
       if (slice_size == 0xffff) {
           printf("%s %d\n", __FUNCTION__, __LINE__);
           return;
       }
       //printf("%d %d\n", mb_x, mb_y);
       uint32_t size;
       uint16_t *y = getY(y_data,mb_x,mb_y,8);
       uint16_t *cb = getC(cb_data,mb_x,mb_y,8);
       uint16_t *cr = getC(cr_data,mb_x,mb_y,8);
       //size = encode_slice(y_data, cb_data, cr_data, mb_x, mb_y, slice_size);
       size = encode_slice(y, cb, cr, 0, 0, slice_size);
       new_slice_table[i] = size;
       //printf("size = %d\n",size);

        mb_x += slice_mb_count;
        if (mb_x == mb_x_max ) {
            slice_mb_count = 1 << log2_desired_slice_size_in_mb ;
            mb_x = 0;
            mb_y++;
                
        }
        //for debug
        //break;


    }

    slice_mb_count = 1 << log2_desired_slice_size_in_mb;
    mb_x = 0;
    mb_y = 0;
    for (i = 0; i < slice_num_max ; i++) {

        while ((mb_x_max - mb_x) < slice_mb_count)
            slice_mb_count >>=1;

        setSliceTalbeFlush(new_slice_table[i], slice_size_table_offset + (i * 2));
        //printf("f table offset %d \n", slice_size_table_offset + (i * 2));

        mb_x += slice_mb_count;
        if (mb_x == mb_x_max ) {
            slice_mb_count = 1 << log2_desired_slice_size_in_mb ;
            mb_x = 0;
            mb_y++;
        }
        //for debug
        //break;


    }
    
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

