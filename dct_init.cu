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
#include "dct_init.h"

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

