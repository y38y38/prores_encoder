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
//#include <string.h>
#include <math.h>

#include "config.h"
#include "prores.h"
#include "dct.h"

void first_dct1(double *in, double *out) {
	double step1[8];
	double step2[8];
	double step3[8];
	double step4[8];

	step1[0] = in[0] + in[7];
	step1[1] = in[1] + in[6];
	step1[2] = in[2] + in[5];
	step1[3] = in[3] + in[4];
	step1[4] = -1 * in[4] + in[3];
	step1[5] = -1 * in[5] + in[2];
	step1[6] = -1 * in[6] + in[1];
	step1[7] = -1 * in[7] + in[0];



	step2[0] = step1[0] + step1[3];
	step2[1] = step1[1] + step1[2];
	step2[2] = (-1) * step1[2] + step1[1];
	step2[3] = (-1) * step1[3] + step1[0];

	
	step2[4] = step1[4];

	step2[5] = (step1[5] * (-1) * cos(M_PI / 4)) + (step1[6] * cos(M_PI/4)) ;
	step2[6] = (step1[6] * cos(M_PI/4)) + (step1[5] * cos(M_PI / 4));

	step2[7] = step1[7];



	step3[0] = (step2[0] * cos(M_PI / 4)) + (step2[1] * cos(M_PI/4));
	step3[1] = (step2[1] * (-1) * cos(M_PI / 4) ) + (step2[0] * cos(M_PI/4));

	step3[2] = (step2[2] * sin(M_PI / 8)) + ( step2[3] * cos(M_PI/8));
	step3[3] = (step2[3] * cos( 3 * M_PI / 8)) + (step2[2] * (-1) * sin( 3 *  M_PI / 8));

	step3[4] = step2[4] + step2[5];
	step3[5] = (-1) * step2[5] + step2[4];
	step3[6] = (-1) * step2[6] + step2[7];
	step3[7] = step2[6] + step2[7];

	step4[0] = step3[0];
	step4[1] = step3[1];
	step4[2] = step3[2];
	step4[3] = step3[3];

	step4[4] = (step3[4] * sin(M_PI / 16)) + (step3[7] * cos(M_PI/16));
	step4[5] = (step3[5] * sin((5 * M_PI) / 16)) + (step3[6] * cos(5 * M_PI/ 16));
	step4[6] = (step3[6] * cos((3 * M_PI) / 16)) + (step3[5] * (-1) * sin((3 * M_PI)  / 16));
	step4[7] = (step3[7] * cos( 7 *  M_PI / 16)) + (step3[4] * (-1) * sin(( 7 * M_PI) / 16));



	double step5[8];
	step5[0] = step4[0];
	step5[1] = step4[4];
	step5[2] = step4[2];
	step5[3] = step4[6];
	step5[4] = step4[1];
	step5[5] = step4[5];
	step5[6] = step4[3];
	step5[7] = step4[7];




	out[0] = step5[0] * 0.5;
	out[1] = step5[1] * 0.5;
	out[2] = step5[2] * 0.5;
	out[3] = step5[3] * 0.5;
	out[4] = step5[4] * 0.5;
	out[5] = step5[5] * 0.5;
	out[6] = step5[6] * 0.5;
	out[7] = step5[7]  * 0.5;
	return;

}


int dct_block_first(int16_t * block) {
	int i,j;
	double in[64];
	double out1[64];
	double out2[64];
	double out3[64];
	for(i=0;i<64;i++) {
		in[i] = (double)block[i];
	}

	for(i=0;i < 64;i+=8) {
		first_dct1(in + i, out1 + i);
	}

	for(i=0;i<8;i++) {
		for(j=0;j<8;j++) {
				out2[i * 8 + j] = out1[j * 8 + i];
		}
	}

	for(i=0;i<64;i+=8) {
		first_dct1(out2 + i, out3 + i);
	}
	for(i=0;i<8;i++) {
		block[(i*8)] = out3[i];
		block[(i*8)+1] = out3[8+i];
		block[(i*8)+2] = out3[16+i];
		block[(i*8)+3] = out3[24+i];
		block[(i*8)+4] = out3[32+i];
		block[(i*8)+5] = out3[40+i];
		block[(i*8)+6] = out3[48+i];
		block[(i*8)+7] = out3[56+i];
	}
	return 0;
}



static double kc_value[MAX_X][MAX_Y][MAX_X][MAX_Y];

int dct_block(int16_t *block) {
	static int first2 = 0;
	if (first2 == 0) {
		aprint_block(block);
		first2 ++;
	}

#ifdef FIRST_DCT_A
	dct_block_first(block);
#else
    int h,v,i,x,y;
	double value;

    double result[MAX_X * MAX_Y];
    for (v=0;v<MAX_Y;v++) {
        for (h=0;h<MAX_X;h++) {
			value = 0;
		    for(y=0;y<MAX_Y;y++) {

        		for(x=0;x<MAX_X;x++) {
#ifdef PRE_CALC_COS
            		value += block[(y * 8) + x] *  kc_value[x][y][h][v];
#else
            		double kc = cos((M_PI * v * ((2.0 * y) + 1.0)) / 16.0) * cos((M_PI * h * ((2.0 * x) + 1.0)) / 16.0);
            		value += block[(y * 8) + x] *  kc;
#endif
        		}
    		}
    		if ( h == 0) {
#ifdef DEL_SQRT //changed quality
#ifdef DEL_DIVISION
        		value *= 0.70710678118;
#else
				//better quality
        		value *= 1/ 1.41421356237;
#endif

#else
        		value *= 1/ sqrt(2.0);
#endif
    		} else {
//        		value *= 1;
    		}
    		if (v == 0) {
#ifdef DEL_SQRT
#ifdef DEL_DIVISION
        		value *= 0.70710678118;
#else
				//better quality
        		value *= 1 / 1.41421356237;
#endif
#else
        		value *= 1/ sqrt(2.0);
#endif
    		} else {
//        		value *= 1;
    		}
			//double can't shift
    		
			
			
			value = value / 4;

            result[(v << 3) + h] = value;
        }
    }
    for(i = 0;i<MAX_X*MAX_Y;i++) {
        block[i] = (int16_t)result[i];
    }
#endif
	static int first = 0;
	if (first == 0) {
		aprint_block(block);
		first ++;
	}

    return 0;
}

void dct_init(void)
{
    int h,v,x,y;
    for (v=0;v<MAX_Y;v++) {
        for (h=0;h<MAX_X;h++) {
		    for(y=0;y<MAX_Y;y++) {
        		for(x=0;x<MAX_X;x++) {
            		kc_value[x][y][h][v] = cos((M_PI * v * ((2.0 * y) + 1.0)) / 16.0) * cos((M_PI * h * ((2.0 * x) + 1.0)) / 16.0);
				}
			}
        }
    }
    return;
}
