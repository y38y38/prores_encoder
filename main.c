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
#include <unistd.h>
#include "encoder.h"
#if 1
#define MATRIX_NUM (64)
uint8_t luma_matrix2_[MATRIX_NUM ];
uint8_t chroma_matrix2_[MATRIX_NUM];
uint32_t qscale_table_size_ = 0;
uint8_t *qscale_table_ = NULL;
uint32_t block_num_ = 0;
uint32_t width_ = 0;
uint32_t height_ = 0;
char *input_file_ = NULL;
char *output_file_ = NULL ;

int32_t Text2Matrix(char *file, uint8_t *matrix, int row_num, int column_num)
{
    
    FILE *input = fopen((char*)file, "r");
    if (input == NULL) {
        printf("%d\n", __LINE__);
        return -1;
    }
    for (int32_t i=0; i<column_num;i++) {
        char temp[1024];
        char * ret = fgets(temp, 1024, input);
        if (ret != temp) {
            printf("%d\n", __LINE__);
            return -1;
        }
        int len = 0;
        for (int32_t j = 0;j<row_num;j++) {
            ret = strstr(temp + len, ",");
            if (ret == NULL) {
                if (j == 7) {
                    ret = strstr(temp + len, "\r");
                    if (ret == NULL) {
                        printf("%d\n", __LINE__);
                        return -1;
                    }
                } else {
                    printf("%d\n", __LINE__);
                    return -1;
                }
            }
            char temp2[1024];
            memset(temp2, 0x0, 1024);
            memcpy(temp2, temp + len , ret - (temp + len));
            len = ret - temp + 1;
            int val = atoi(temp2);
            matrix[(i*8) + j] = val;
            printf("%d ",val);
        }
            printf("\n");
    }
    return 0;

}
int32_t SetChromaMatrix(char *matrix_file)
{
    return Text2Matrix(matrix_file, chroma_matrix2_, 8, 8);
}

int32_t SetLumaMatrix(char *matrix_file)
{
    return Text2Matrix(matrix_file, luma_matrix2_, 8, 8);
}
int32_t SetQscaleTable(char *qscale_file, int32_t table_size)
{

    qscale_table_ = (uint8_t*)malloc(table_size * sizeof(uint8_t));
    if (qscale_table_ == NULL) {
        printf("malloc %d\n", __LINE__);
        return -1;
    }
    return Text2Matrix(qscale_file, qscale_table_ , 1, table_size);
}
int32_t GetParam(int argc, char **argv)
{
    char *luma_matrix_file = NULL;
    char *chroma_matrix_file = NULL;
    char *qscale_file = NULL;
    char *input_file = NULL;
    char *output_file = NULL;
    int opt;
    int32_t ret;
    while((opt = getopt(argc, argv, "l:c:q:w:h:i:o:m:")) != -1) {
        switch(opt) {
            case 'l':
                luma_matrix_file = optarg;
                break;
            case 'c':
                chroma_matrix_file = optarg;
                break;
            case 'q':
                qscale_file = optarg;
                break;
            case 'w':
                width_ = atoi(optarg);
                break;
            case 'h':
                height_ = atoi(optarg);
                break;
            case 'i':
                input_file = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'm':
                block_num_ = atoi(optarg);
                break;
            default:
                printf("error %d\n", __LINE__);
                return -1;
        }
    }
    printf("l %s\n",luma_matrix_file);
    printf("c %s\n",chroma_matrix_file);
    printf("q %s\n",qscale_file);
    printf("i %s\n",input_file);
    printf("o %s\n",output_file);

    if (luma_matrix_file != NULL) {
        ret =  SetLumaMatrix(luma_matrix_file);
        if (ret < 0) {
            printf("err %d\n", __LINE__);
            return -1;
        }
    } else {
        printf("err %d\n", __LINE__);
        return -1;
    }
    if (chroma_matrix_file != NULL) {
        ret = SetChromaMatrix(chroma_matrix_file);
        if (ret < 0) {
            printf("err %d\n", __LINE__);
            return -1;
        }
    } else {
        printf("err %d\n", __LINE__);
        return -1;
    }
    if (width_ == 0) {
        width_ = 128;
    }
    if (height_ == 0) {
        height_ = 16;
    }
    if (block_num_  == 0) {
        block_num_   = 8;
    }

    qscale_table_size_  = GetSliceNum(width_, height_, block_num_);
    printf("slize num %d\n", qscale_table_size_);

    if (qscale_file != NULL)  {
        ret = SetQscaleTable(qscale_file,qscale_table_size_);
        if (ret < 0) {
            printf("err %d\n", __LINE__);
            return -1;
        }
    } else {
        printf("err %d\n", __LINE__);
        return -1;
    }
    if (input_file != NULL) {
        input_file_ =  input_file;
    } else {
        printf("err %d\n", __LINE__);
        return -1;
    }
    if (output_file != NULL) {
        output_file_ =  output_file;
    } else {
        printf("err %d\n", __LINE__);
        return -1;
    }

    return 0;

}
#endif
/*
 * ./encoder [-l luma_matrix_file] [-c  chroma_matrix_file] [-q qscale_file] [-w width] [-h height] [-m block_num_of_macroblock] -i input_file -o output_file
 */
int main(int argc, char **argv)
{
    int32_t ret = GetParam(argc, argv);
    if (ret < 0) {
        printf("error %d\n", __LINE__);
        return -1;
    }
    //FILE *input = fopen(argv[1], "r");
    FILE *input = fopen(input_file_, "r");
    if (input == NULL) {
        printf("err %s\n", input_file_);
        return -1;
    }
    FILE *output = fopen(output_file_, "w");
    if (output == NULL) {
        printf("err %s\n", output_file_);
        return -1;
    }
    //decode_init();
    uint32_t size = width_*height_ * 2;
    uint16_t *y_data = (uint16_t*)malloc(size);
    if (y_data == NULL) {
        printf("%d\n", __LINE__);
        return 0;
    }
    uint16_t *cb_data = (uint16_t*)malloc(size/2);
    if (cb_data == NULL) {
        printf("%d\n", __LINE__);
        return 0;
    }
    uint16_t *cr_data = (uint16_t*)malloc(size/2);
    if (cr_data == NULL) {
        printf("%d\n", __LINE__);
        return 0;
    }
    struct encoder_param param;
    param.luma_matrix = luma_matrix2_;
    param.chroma_matrix = chroma_matrix2_;
    param.qscale_table_size = qscale_table_size_;
    param.qscale_table = qscale_table_;
    param.block_num = block_num_;
    param.width  = width_;
    param.heigth = height_;
    param.y_data = y_data;
    param.cb_data = cb_data;
    param.cr_data = cr_data;

    encoder_init();
    for (int32_t i=0;;i++) {
        size_t readsize = fread(y_data, 1, size, input);
        if (readsize != size) {
            printf("%d %d\n", __LINE__, (int32_t)readsize);
            break;
        }
        readsize = fread(cb_data, 1, (size /2), input);
        if (readsize != (size / 2)) {
            printf("%d\n", __LINE__);
            break;
        }
        readsize = fread(cr_data, 1, (size /2), input);
        if (readsize != (size / 2)) {
            printf("%d\n", __LINE__);
            break;
        }
        uint32_t frame_size;
        uint8_t *frame = encode_frame(&param, &frame_size);

        printf("frame size %d\n", frame_size);

        size_t writesize = fwrite(frame, 1, frame_size,  output);
        if (writesize != frame_size) {
            printf("%s %d %d\n", __FUNCTION__, __LINE__, (int)writesize);
            //printf("write %d %p %d %p \n", (int)writesize, raw_data, raw_size,output);
            return -1;
        }
        if (i==0) {
          break;
        }
        //printf("end frame\n");
        printf(".");
    }
    free(y_data);
    free(cb_data);
    free(cr_data);
    fclose(input);
    fclose(output);

    return 0;
}

