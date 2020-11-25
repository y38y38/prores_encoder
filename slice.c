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

#include "config.h"
#include "dct.h"
#include "bitstream.h"
#include "encoder.h"
#include "slice.h"


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
        

}
void print_slice(int16_t *slice, int32_t mb_num)
{
    int32_t i;
    for(i=0;i<mb_num;i++) {
        print_mb(slice + (i*64)*4);
    }
        

}
void print_pixels(int16_t *slice, int32_t mb_num)
{
    int32_t i;
    for(i=0;i<mb_num*4*64;i++) {
        printf("%d\n", slice[i]);
    }
        

}


int32_t GetAbs(int32_t val)
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
#ifdef DEL_DIVISION
    int32_t q  = val >> k;
#else
    int32_t q  = floor( val / pow(2,k));
#endif

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
#ifdef DEL_DIVISION
        uint32_t tmp = (k==0) ? 1 : (2<<(k-1));
        uint32_t r = val & (tmp -1 );
#else
        uint32_t tmp = pow(2,k);
        uint32_t r = val % tmp;
#endif
        uint32_t codeword = (1 << k) | r;
        setBit(codeword, q + 1 + k );
        //printf("2: %x %d\n",codeword, q+1+k);
    }
    return;
}
void exp_golomb_code(int32_t k, uint32_t val)
{
#ifdef DEL_DIVISION
    int32_t q = floor(log2(val + ((k==0) ? 1 : (2<<(k-1))))) - k;
#else
    int32_t q = floor(log2(val + pow(2, k))) - k;
#endif


    //printf("pow %f %d ", pow(2,k),val);
#ifdef DEL_DIVISION
    uint32_t sum = val + ((k==0) ? 1 : (2<<(k-1)));
#else
    uint32_t sum = val + pow(2, k);
#endif
    int32_t codeword_length = (2 * q) + k + 1;

    setBit(sum, codeword_length);
    //printf("3: val:%x q:%d k:%d bits:%d k:%d\n", sum,q, k, codeword_length,k );
    return;
}
void rice_exp_combo_code(int32_t last_rice_q, int32_t k_rice, int32_t k_exp, uint32_t val)
{
#ifdef DEL_DIVISION
//    uint32_t value = (last_rice_q + 1) * ((k_rice==0) ? 1 : (2<<(k_rice-1)));
    uint32_t value = (last_rice_q + 1) << k_rice;
#else
    uint32_t value = (last_rice_q + 1) * pow(2, k_rice);
#endif

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
        sn = 2 * GetAbs(val);

    } else {
        sn = 2 * GetAbs(val) - 1;
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
        abs_previousDCDiff = GetAbs(previousDCDiff );
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

                abs_level_minus_1 = GetAbs(level) - 1;
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
#define MB_IN_BLOCK                   (4)
#define MB_422C_IN_BLCCK              (2)
#define BLOCK_IN_PIXEL               (64)
#define MB_HORIZONTAL_Y_IN_PIXEL     (16)
#define MB_HORIZONTAL_422C_IN_PIXEL   (8)
#define MB_VERTIVAL_IN_PIXEL         (16)
#define MAX_MB_SIZE_IN_MB             (8)
#define BLOCK_HORIZONTAL_IN_PIXEL     (8)
#define BLOCK_VERTIVAL_IN_PIXEL       (8)


void getYDataToBlock(uint16_t *dst, uint16_t*src,uint32_t mb_x, uint32_t mb_y, uint32_t mb_size)
{
    uint16_t pixel_data[MAX_MB_SIZE_IN_MB * MB_IN_BLOCK* BLOCK_IN_PIXEL * sizeof(uint16_t)];
    int32_t i;
    for (i=0;i<MB_VERTIVAL_IN_PIXEL;i++) {
        memcpy(pixel_data + (i * (MB_HORIZONTAL_Y_IN_PIXEL* mb_size)), 
               src + (mb_x * MB_HORIZONTAL_Y_IN_PIXEL)  + ((mb_y * MB_VERTIVAL_IN_PIXEL) * mb_size * MB_HORIZONTAL_Y_IN_PIXEL) + (i * mb_size * MB_HORIZONTAL_Y_IN_PIXEL), 
               MB_HORIZONTAL_Y_IN_PIXEL* mb_size * sizeof(uint16_t));
    }
    int32_t vertical;
    int32_t block;
    int32_t block_position = 0;
    for (i=0;i<mb_size;i++) {
        for (block = 0 ; block < MB_IN_BLOCK;block++) {
            for(vertical= 0;vertical<BLOCK_VERTIVAL_IN_PIXEL;vertical++) {
                if (block == 0) {
                    block_position = 0;
                } else if (block == 1) {
                    block_position =  BLOCK_HORIZONTAL_IN_PIXEL;
                } else if (block == 2) {
                    block_position =  mb_size * MB_HORIZONTAL_Y_IN_PIXEL* BLOCK_VERTIVAL_IN_PIXEL;
                } else {
                    block_position =  (mb_size * MB_HORIZONTAL_Y_IN_PIXEL* BLOCK_VERTIVAL_IN_PIXEL) + BLOCK_HORIZONTAL_IN_PIXEL;
                }
                memcpy(dst + (i * (MB_HORIZONTAL_Y_IN_PIXEL * MB_VERTIVAL_IN_PIXEL)) +  (block * BLOCK_IN_PIXEL) + (vertical * BLOCK_HORIZONTAL_IN_PIXEL) , 
                        pixel_data + ( MB_HORIZONTAL_Y_IN_PIXEL * i) + block_position + vertical * (mb_size * MB_HORIZONTAL_Y_IN_PIXEL),
                        BLOCK_HORIZONTAL_IN_PIXEL * sizeof(uint16_t));
            }
        }

    }
    return;
}

void getCbDataToBlock(uint16_t *dst, uint16_t*src,uint32_t mb_x, uint32_t mb_y, uint32_t mb_size)
{
    //test_data(cb_data);
    int32_t i;
    //for (i=0;i<16;i++) {

    uint16_t pixel_data[MAX_MB_SIZE_IN_MB * MB_IN_BLOCK * BLOCK_IN_PIXEL * sizeof(uint16_t)];
    for (i=0;i<MB_VERTIVAL_IN_PIXEL;i++) {
        memcpy(pixel_data + (i * (MB_HORIZONTAL_422C_IN_PIXEL * mb_size)), 
               src + (mb_x * MB_HORIZONTAL_422C_IN_PIXEL)  + ((mb_y * MB_VERTIVAL_IN_PIXEL) * mb_size*MB_HORIZONTAL_422C_IN_PIXEL ) + (i * (mb_size*MB_HORIZONTAL_422C_IN_PIXEL )), 
               MB_HORIZONTAL_422C_IN_PIXEL * mb_size * sizeof(uint16_t));
    }

    //memset(pixel_data, 0x0, 64);
    int32_t vertical;
    int32_t block;
    int32_t block_position = 0;
    //printf("aa %p\n", pixel_data);
    for (i=0;i<mb_size;i++) {
        for (block = 0 ; block < MB_422C_IN_BLCCK;block++) { //4:2:2
            for(vertical= 0;vertical<BLOCK_VERTIVAL_IN_PIXEL;vertical++) {
                if (block == 0) {
                    block_position = 0;
                } else  {
                    block_position =  mb_size * MB_HORIZONTAL_422C_IN_PIXEL * BLOCK_VERTIVAL_IN_PIXEL;
                }
                memcpy(dst + (i * (MB_HORIZONTAL_422C_IN_PIXEL * MB_VERTIVAL_IN_PIXEL)) +  (block * BLOCK_IN_PIXEL) + (vertical * BLOCK_HORIZONTAL_IN_PIXEL) , 
                        pixel_data + ( MB_HORIZONTAL_422C_IN_PIXEL  * i) + block_position + vertical * (mb_size * MB_HORIZONTAL_422C_IN_PIXEL),
                        BLOCK_HORIZONTAL_IN_PIXEL * sizeof(uint16_t));
                //printf("%x %d ", pixel_data[0], ( 8 * i) + block_position + vertical * (mb_size * 8));
            }
        }

    }
    return;



}
void getYver2block(uint16_t *out, uint16_t *in, uint32_t x, uint32_t y, int32_t horizontal, int32_t vertical)
{
	//printf("%d %d %d %d\n", x,y, horizontal, vertical);
	int i;
	for(i=0;i<8;i++) {
		memcpy(out + (i*8),
		in + x + (horizontal * y) + (horizontal * i),
		8 * sizeof(uint16_t));
	//	printf(" %x\n", *(uint16_t*)(in + x + (horizontal * y) + (horizontal * i)));
	}
}
//get 1 slice data
void getYver2(uint16_t *out, uint16_t *in, uint32_t mb_x, uint32_t mb_y, int32_t mb_size, int32_t horizontal, int32_t vertical)
{
	int i;
    int32_t block;
	int offset_x,offset_y;
    for (i=0;i<mb_size;i++) {
        for (block = 0 ; block < MB_IN_BLOCK;block++) {
			if (block == 0) {
				offset_x = 0;
				offset_y = 0;
			} else if (block == 1) {
				offset_x = 8;
				offset_y = 0;
			} else if (block == 2) {
				offset_x = 0;
				offset_y = 8;
			} else {
				offset_x = 8;
				offset_y = 8;
			}
			getYver2block(out  +  i * 64 * 4 + (block * 64), in, (mb_x * 16) + (i * 16) + offset_x, (mb_y * 16) + offset_y, horizontal, vertical);
        }

    }
	return;
}

void getCver2block(uint16_t *out, uint16_t *in, uint32_t x, uint32_t y, int32_t horizontal, int32_t vertical)
{
	//printf("%d %d %d %d\n", x,y, horizontal, vertical);
	int i;
	for(i=0;i<8;i++) {
		memcpy(out + (i*8),
		in + x + (horizontal * y) + (horizontal * i),
		8 * sizeof(uint16_t));
		//printf(" %x\n", *(uint16_t*)(in + x + (horizontal * y) + (horizontal * i)));
	}
}
//get 1 slice data
void getCver2(uint16_t *out, uint16_t *in, uint32_t mb_x, uint32_t mb_y, int32_t mb_size, int32_t horizontal, int32_t vertical)
{
	int i;
    int32_t block;
	int offset_x,offset_y;
    for (i=0;i<mb_size;i++) {
        for (block = 0 ; block < MB_422C_IN_BLCCK;block++) {
			if (block == 0) {
				offset_x = 0;
				offset_y = 0;
			} else {
				offset_x = 0;
				offset_y = 8;
			}
			getCver2block(out  +  (i * 64 * 2) + (block * 64), in, (mb_x * 8) + (i * 8) + offset_x, (mb_y * 16) + offset_y, horizontal>>1, vertical);
        }

    }
	return;
}


void encode_qt(int16_t *block, uint8_t *qmat, int32_t  block_num)
{

    int16_t *data;
    int32_t i,j;
    for (i = 0; i < block_num; i++) {
        data = block + (i * BLOCK_IN_PIXEL);
        for (j=0;j<BLOCK_IN_PIXEL;j++) {
            data[j] = data [j] / ( qmat[j]) ;
        }

    }
}
void encode_qscale(int16_t *block, uint8_t scale, int32_t  block_num)
{

    int16_t *data;
    int32_t i,j;
    for (i = 0; i < block_num; i++) {
        data = block + (i*BLOCK_IN_PIXEL);
        for (j=0;j<BLOCK_IN_PIXEL;j++) {
            data[j] = data [j] / scale;
        }

    }
}
void pre_quant(int16_t *block, int32_t  block_num)
{

    int16_t *data;
    int32_t i,j;
    for (i = 0; i < block_num; i++) {
        data = block + (i*BLOCK_IN_PIXEL);
        for (j=0;j<BLOCK_IN_PIXEL;j++) {
            data[j] = data [j] << 3;
        }

    }
}
void pre_dct(int16_t *block, int32_t  block_num)
{

    int16_t *data;
    int32_t i,j;
    for (i = 0; i < block_num; i++) {
        data = block + (i*BLOCK_IN_PIXEL);
        for (j=0;j<BLOCK_IN_PIXEL;j++) {
            data[j] = (data[j] >> 1) - 256;
        }

    }
}
// macro block num * block num per macro  block * pixel num per block * pixel size
// (mb_size(8) * MB_IN_BLOCK(4) * BLOCK_IN_PIXEL(64)
#define MAX_SLICE_DATA	(2048)

int16_t y_slice[MAX_SLICE_DATA];
int16_t cb_slice[MAX_SLICE_DATA];
int16_t cr_slice[MAX_SLICE_DATA];

uint32_t encode_slice_y(uint16_t*y_data, uint32_t mb_x, uint32_t mb_y, int32_t scale, uint8_t *matrix, uint32_t slice_size_in_mb, int horizontal, int vertical)
{
    //print_slice(y_data, 8);
    //printf("%s start\n", __FUNCTION__);
    uint32_t start_offset= getBitSize();
    //printf("start_offset %d\n", start_offset);

//    getYDataToBlock((uint16_t*)y_slice, y_data,mb_x,mb_y,slice_size_in_mb);
#if 0
	static int first = 0;
	if (first == 0) {
		FILE* output = fopen("aa.yuv", "w");
		fwrite(y_slice, 1, 8 * 16 * 16 * 2, output);
		fclose(output);
		first = 1;
	}
#endif
	getYver2((uint16_t*)y_slice, y_data, mb_x,mb_y,slice_size_in_mb, horizontal, vertical);

    //printf("before\n");
    //print_pixels(y_slice, 8);

    //printf("pre\n");
    pre_dct(y_slice, slice_size_in_mb * MB_IN_BLOCK);
    //printf("after\n");
    //print_pixels(y_slice, 8);

    int32_t i;
    //print_slice(y_slice, 8);
    for (i = 0;i< slice_size_in_mb * MB_IN_BLOCK;i++) {
        dct_block(&y_slice[i * BLOCK_IN_PIXEL]);
    }
    //print_slice(y_slice, 8);
    //print_slice(y_slice, 8);
    //

    pre_quant(y_slice, slice_size_in_mb * MB_IN_BLOCK);

    encode_qt(y_slice, matrix, slice_size_in_mb * MB_IN_BLOCK);
    //print_slice(y_slice, 8);
    encode_qscale(y_slice, scale , slice_size_in_mb * MB_IN_BLOCK);


    entropy_encode_dc_coefficients(y_slice, slice_size_in_mb * MB_IN_BLOCK);
    entropy_encode_ac_coefficients(y_slice, slice_size_in_mb * MB_IN_BLOCK);

    //byte aliened
    uint32_t size  = getBitSize();
    if (size & 7 )  {
        setBit(0x0, 8 - (size & 7));
    }
    uint32_t current_offset = getBitSize();
    //printf("current_offset %d\n",current_offset );
    //printf("%s end\n", __FUNCTION__);
    return ((current_offset - start_offset)/8);
}
uint32_t encode_slice_cb(uint16_t*cb_data, uint32_t mb_x, uint32_t mb_y, int32_t scale, uint8_t *matrix, uint32_t slice_size_in_mb, int horizontal, int vertical)
{
    //printf("cb start\n");
    uint32_t start_offset= getBitSize();

    //getCbDataToBlock((uint16_t*)cb_slice, cb_data,mb_x,mb_y,slice_size_in_mb);
	getCver2((uint16_t*)cb_slice, cb_data, mb_x,mb_y,slice_size_in_mb, horizontal, vertical);

    pre_dct(cb_slice, slice_size_in_mb * MB_422C_IN_BLCCK);

    int32_t i;
    //memset(cb_slice, 0x0, 64);
    //extern int g_first;
    //g_first = 0;
    //print_slice_cb(cb_slice, 4);
    for (i = 0;i< slice_size_in_mb * MB_422C_IN_BLCCK;i++) {
        dct_block(&cb_slice[i* BLOCK_IN_PIXEL]);
    }
    //printf("af\n");
    //print_slice_cb(cb_slice, 4);
    pre_quant(cb_slice, slice_size_in_mb * MB_422C_IN_BLCCK);

    encode_qt(cb_slice, matrix, slice_size_in_mb * MB_422C_IN_BLCCK);
    encode_qscale(cb_slice,scale , slice_size_in_mb * MB_422C_IN_BLCCK);

    entropy_encode_dc_coefficients(cb_slice, slice_size_in_mb * MB_422C_IN_BLCCK);
    entropy_encode_ac_coefficients(cb_slice, slice_size_in_mb * MB_422C_IN_BLCCK);

    //byte aliened
    uint32_t size  = getBitSize();
    if (size & 0x7 )  {
        setBit(0x0, 8 - (size % 8));
    }
    uint32_t current_offset = getBitSize();
    //printf("%s end\n", __FUNCTION__);
    return ((current_offset - start_offset)/8);
}
uint32_t encode_slice_cr(uint16_t*cr_data, uint32_t mb_x, uint32_t mb_y, int32_t scale, uint8_t *matrix, uint32_t slice_size_in_mb, int horizontal, int vertical)
{
    //printf("%s start\n", __FUNCTION__);
    uint32_t start_offset= getBitSize();

//    getCbDataToBlock((uint16_t*)cr_slice, cr_data,mb_x,mb_y,slice_size_in_mb);
	getCver2((uint16_t*)cr_slice, cr_data, mb_x,mb_y,slice_size_in_mb, horizontal, vertical);

    pre_dct(cr_slice, slice_size_in_mb * MB_422C_IN_BLCCK);

    int32_t i;
    //extern int32_t g_first;
    //g_first = 0;
    //print_slice_cb(cr_slice, 4);
    for (i = 0;i< slice_size_in_mb * MB_422C_IN_BLCCK;i++) {
        dct_block(&cr_slice[i* BLOCK_IN_PIXEL]);
    }
    //print_slice_cb(cr_slice, 4);
    pre_quant(cr_slice, slice_size_in_mb * MB_422C_IN_BLCCK);
    encode_qt(cr_slice, matrix, slice_size_in_mb * MB_422C_IN_BLCCK);
    encode_qscale(cr_slice,scale , slice_size_in_mb * MB_422C_IN_BLCCK);

    entropy_encode_dc_coefficients(cr_slice, slice_size_in_mb * MB_422C_IN_BLCCK);
    entropy_encode_ac_coefficients(cr_slice, slice_size_in_mb * MB_422C_IN_BLCCK);
    //byte aliened
    uint32_t size  = getBitSize();
    if (size & 7 )  {
        setBit(0x0, 8 - (size % 8));
    }
    uint32_t current_offset = getBitSize();
    //printf("%s end\n", __FUNCTION__);
    return ((current_offset - start_offset)/8);
}
uint8_t qScale2quantization_index(uint8_t qscale)
{
    return qscale;
}
uint32_t encode_slice(struct Slice *param)
{
    uint32_t start_offset= getBitSize();

    uint8_t slice_header_size = 6;

    setBit(slice_header_size , 5);

    uint8_t reserve =0x0;
    setBit(reserve, 3);

    setByte(&param->qscale, 1);

    uint32_t code_size_of_y_data_offset = getBitSize();
    code_size_of_y_data_offset = code_size_of_y_data_offset >> 3;
    uint16_t size = 0;
    uint16_t coded_size_of_y_data = SET_DATA16(size);
    setByte((uint8_t*)&coded_size_of_y_data , 2);

    uint32_t code_size_of_cb_data_offset = getBitSize();
    code_size_of_cb_data_offset = code_size_of_cb_data_offset >> 3 ;
    size = 0;
    uint16_t coded_size_of_cb_data = SET_DATA16(size);
    setByte((uint8_t*)&coded_size_of_cb_data , 2);



    //printf("%s start\n", __FUNCTION__);

    size = (uint16_t)encode_slice_y(param->y_data, param->mb_x, param->mb_y, param->qscale, param->luma_matrix, param->slice_size_in_mb, param->horizontal, param->vertical);
    //exit(1);
    uint16_t y_size  = SET_DATA16(size);
    //printf("y %d %x\n", size, size);
    
    uint16_t cb_size;
    if (param->format_444 == true) {
        size = (uint16_t)encode_slice_y(param->cb_data, param->mb_x, param->mb_y, param->qscale, param->chroma_matrix, param->slice_size_in_mb, param->horizontal, param->vertical);
        cb_size = SET_DATA16(size);
        //printf("cb %d\n", size);
        //exit(1);
        size = (uint16_t)encode_slice_y(param->cr_data, param->mb_x, param->mb_y, param->qscale, param->chroma_matrix, param->slice_size_in_mb, param->horizontal, param->vertical);
    } else {
        size = (uint16_t)encode_slice_cb(param->cb_data, param->mb_x, param->mb_y, param->qscale, param->chroma_matrix, param->slice_size_in_mb, param->horizontal, param->vertical);
        cb_size = SET_DATA16(size);
        //printf("cb %d\n", size);
        //exit(1);
        size = (uint16_t)encode_slice_cr(param->cr_data, param->mb_x, param->mb_y, param->qscale, param->chroma_matrix, param->slice_size_in_mb, param->horizontal, param->vertical);
    }

    //uint16_t cr_size = SET_DATA16(size);
    //printf("cr%d\n", size);
    setByteInOffset(code_size_of_y_data_offset , (uint8_t *)&y_size, 2);
    //printf("%d %x\n",code_size_of_y_data_offset,code_size_of_y_data_offset); 
    setByteInOffset(code_size_of_cb_data_offset , (uint8_t *)&cb_size, 2);
    uint32_t current_offset = getBitSize();
    return ((current_offset - start_offset)/8);
}

