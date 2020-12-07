
/**
 *
 * Copyright (c) 2020 Yuusuke Miyazaki
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 *
 **/

#ifndef __PRORES__H__
#define __PRORES__H__

#define MAX_X   (8)
#define MAX_Y   (8)

#define MATRIX_ROW_NUM  (8)
#define MATRIX_COLUMN_NUM  (8)
#define MATRIX_NUM (MATRIX_ROW_NUM*MATRIX_COLUMN_NUM)

#define MB_IN_BLOCK                   (4)
#define MB_422C_IN_BLCCK              (2)
#define BLOCK_IN_PIXEL               (64)
#define MB_HORIZONTAL_Y_IN_PIXEL     (16)
#define MB_HORIZONTAL_422C_IN_PIXEL   (8)
#define MB_VERTIVAL_IN_PIXEL         (16)
#define MAX_MB_SIZE_IN_MB             (8)
#define BLOCK_HORIZONTAL_IN_PIXEL     (8)
#define BLOCK_VERTIVAL_IN_PIXEL       (8)


#endif
