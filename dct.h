/**
 *
 * Copyright (c) 2020 Yuusuke Miyazaki
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 *
 **/
#ifndef __DCT_H__
#define __DCT_H__

#define		GET_KC_INDEX(x,y,h,v)	\
			(x * 8 * 8 * 8)  + \
			(y * 8 * 8) + \
			(h * 8) + \
			(v)

#define		KC_INDEX_MAX		(8*8*8*8)

extern void dct_init(double *kc_value);
extern int dct_block(int16_t *block, double *kc_value);
#endif 
