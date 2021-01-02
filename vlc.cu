
/**
 *
 * Copyright (c) 2020 Yuusuke Miyazaki
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 *
 **/

#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "prores.h"
#include "bitstream.h"
#include "vlc.h"

#define MAX_COEFFICIENT_NUM_PER_BLOCK (64)

static void golomb_rice_code(int32_t k, uint32_t val, struct bitstream *bitstream)
{
    int32_t q  = val >> k;

    if (k ==0) {
        if (q != 0) {
            setBit(bitstream, 0,q);
        }
        setBit(bitstream, 1,1);
    } else {
        uint32_t tmp = (k==0) ? 1 : (2<<(k-1));
        uint32_t r = val & (tmp -1 );

        uint32_t codeword = (1 << k) | r;
        setBit(bitstream, codeword, q + 1 + k );
    }
    return;
}
static void exp_golomb_code(int32_t k, uint32_t val, struct bitstream *bitstream)
{

	//LOG
    int32_t q = floor(log2(val + ((k==0) ? 1 : (2<<(k-1))))) - k;

    uint32_t sum = val + ((k==0) ? 1 : (2<<(k-1)));

    int32_t codeword_length = (2 * q) + k + 1;

    setBit(bitstream, sum, codeword_length);
    return;
}
static void rice_exp_combo_code(int32_t last_rice_q, int32_t k_rice, int32_t k_exp, uint32_t val, struct bitstream *bitstream)
{
    uint32_t value = (last_rice_q + 1) << k_rice;

    if (val < value) {
        golomb_rice_code(k_rice, val, bitstream);
    } else {
        setBit(bitstream, 0,last_rice_q + 1);
        exp_golomb_code(k_exp, val - value, bitstream);
    }
    return;
}
static void entropy_encode_dc_coefficient(bool first, int32_t abs_previousDCDiff , int val, struct bitstream *bitstream)
{
    if (first) {
        exp_golomb_code(5, val, bitstream);
    } else if (abs_previousDCDiff == 0) {
        exp_golomb_code(0, val, bitstream);
    } else if (abs_previousDCDiff == 1) {
        exp_golomb_code(1, val, bitstream);
    } else if (abs_previousDCDiff == 2) {
        rice_exp_combo_code(1,2,3, val, bitstream);
    } else {
        exp_golomb_code(3, val, bitstream);
    }
    return;

}
static void encode_vlc_codeword_ac_run(int32_t previousRun, int32_t val, struct bitstream *bitstream)
{
    if ((previousRun== 0)||(previousRun== 1)) {
        rice_exp_combo_code(2,0,1, val,bitstream);
    } else if ((previousRun== 2)||(previousRun== 3)) {
        rice_exp_combo_code(1,0,1, val, bitstream);
    } else if (previousRun== 4) {
        exp_golomb_code(0, val, bitstream);
    } else if ((previousRun>= 5) && (previousRun <= 8))  {
        rice_exp_combo_code(1,1,2, val,bitstream);
    } else if ((previousRun>= 9) && (previousRun <= 14))  {
        exp_golomb_code(1, val,bitstream);
    } else {
        exp_golomb_code(2, val,bitstream);
    }
    return;

}
static void encode_vlc_codeword_ac_level(int32_t previousLevel, int32_t val, struct bitstream *bitstream)
{
    if (previousLevel== 0) {
        rice_exp_combo_code(2,0,2, val, bitstream);
    } else if (previousLevel== 1) {
        rice_exp_combo_code(1,0,1, val, bitstream);
    } else if (previousLevel== 2) {
        rice_exp_combo_code(2,0,1, val, bitstream);
    } else if (previousLevel == 3)  {
        exp_golomb_code(0, val, bitstream);
    } else if ((previousLevel>= 4) && (previousLevel<= 7))  {
        exp_golomb_code(1, val, bitstream);
    } else {
        exp_golomb_code(2, val, bitstream);
    }
    return;

}

static int32_t GetAbs(int32_t val)
{
    if (val < 0) {
        //printf("m\n");
        return val * -1;
    } else {
        //printf("p\n");
        return val;
    }
}



static int32_t Signedintegertosymbolmapping(int32_t val)
{
    uint32_t sn;
    if (val >=0 ) {
        sn = GetAbs(val) << 1;

    } else {
        sn = (GetAbs(val) << 1) - 1;
    }
    return sn;
}
void entropy_encode_dc_coefficients(int16_t*coefficients, int32_t numBlocks, struct bitstream *bitstream)
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
    entropy_encode_dc_coefficient(true, 0, val, bitstream);

    
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
        entropy_encode_dc_coefficient(false, abs_previousDCDiff, val, bitstream);
        previousDCDiff = DcCoeff - previousDCCoeff;
        previousDCCoeff= DcCoeff;

    }
    return ;
}

//from figure 4
static const uint8_t block_pattern_scan_table[64] = {
     0,  1,  4,  5, 16, 17, 21, 22,
     2,  3,  6,  7, 18, 20, 23, 28,
     8,  9, 12, 13, 19, 24, 27, 29,
    10, 11, 14, 15, 25, 26, 30, 31,
    32, 33, 37, 38, 45, 46, 53, 54,
    34, 36, 39, 44, 47, 52, 55, 60,
    35, 40, 43, 48, 51, 56, 59, 61,
    41, 42, 49, 50, 57, 58, 62, 63,
};
static uint8_t block_pattern_scan_read_order_table[64];


uint32_t entropy_encode_ac_coefficients(int16_t*coefficients, int32_t numBlocks, struct bitstream *bitstream)
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
                encode_vlc_codeword_ac_run(previousRun, run, bitstream);

                abs_level_minus_1 = GetAbs(level) - 1;
                encode_vlc_codeword_ac_level( previousLevelSymbol, abs_level_minus_1, bitstream);
                if (level >=0) {
                    setBit(bitstream, 0,1);
                } else {
                    setBit(bitstream, 1,1);
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

void vlc_init(void)
{
    int32_t i,j;
    for(i=0;i<MATRIX_NUM ;i++) {
        for(j=0;j<MATRIX_NUM ;j++) {
            if (i == block_pattern_scan_table[j]  ) {
                block_pattern_scan_read_order_table[i] = j;
                break;
            }
        }
    }

}