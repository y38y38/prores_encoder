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

#include "config.h"
#include "prores.h"
#include "dct.h"




//static double kc_value[MAX_X][MAX_Y][MAX_X][MAX_Y];

int dct_block(int16_t *block, double *kc_value) {
    int h,v,i,x,y;
	double value;

    double result[MAX_X * MAX_Y];
    for (v=0;v<MAX_Y;v++) {
        for (h=0;h<MAX_X;h++) {
			value = 0;
		    for(y=0;y<MAX_Y;y++) {

        		for(x=0;x<MAX_X;x++) {
            		value += block[(y * 8) + x] *  kc_value[GET_KC_INDEX(x,y,h,v)];
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
    return 0;
}

void dct_init(double *kc_value)
{
    int h,v,x,y;
    for (v=0;v<MAX_Y;v++) {
        for (h=0;h<MAX_X;h++) {
		    for(y=0;y<MAX_Y;y++) {
        		for(x=0;x<MAX_X;x++) {
            		kc_value[GET_KC_INDEX(x,y,h,v)] = cos((M_PI * v * ((2.0 * y) + 1.0)) / 16.0) * cos((M_PI * h * ((2.0 * x) + 1.0)) / 16.0);
				}
			}
        }
    }
    return;
}
