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
uint32_t qscale_table_size_;
uint8_t *qscale_table_;
uint32_t block_num_;
uint32_t width_;
uint32_t heigth_;
int8_t *input_file_;
int8_t *output_file_;

int32_t Text2Matrix(char *file, uint8_t *matrix)
{
    
    FILE *input = fopen((char*)file, "r");
    if (input == NULL) {
        printf("%d\n", __LINE__);
        return -1;
    }
    for (int32_t i=0; i<8;i++) {
        char temp[1024];
        char * ret = fgets(temp, 1024, input);
        if (ret != temp) {
            printf("%d\n", __LINE__);
            return -1;
        }
        int len = 0;
        for (int32_t j = 0;j<8;j++) {
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
            memcpy(temp2, temp, ret - (temp + len));
            len = ret - temp + 1;
            int val = atoi(temp2);
            matrix[(i*8) + j] = val;
        }
    }
    return 0;

}
int32_t SetChromaMatrix(char *matrix_file)
{
    return Text2Matrix(matrix_file, chroma_matrix2_);
}

int32_t SetLumaMatrix(char *matrix_file)
{
    return Text2Matrix(matrix_file, luma_matrix2_);
}
int32_t GetParam(int argc, char **argv)
{
    char *luma_matrix_file = NULL;
    char *chroma_matrix_file = NULL;
    char *qscale_file = NULL;
    char *width = NULL;
    char *height = NULL;
    char *input_file = NULL;
    char *output_file = NULL;
    char *block_num = NULL;
    int opt;
    while((opt = getopt(argc, argv, "l:c:q:w:h:i:o:")) != -1) {
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
                width = optarg;
                break;
            case 'h':
                height = optarg;
                break;
            case 'i':
                input_file = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'm':
                block_num = optarg;
                break;
            default:
                printf("error %d\n", __LINE__);
                return 1;
        }
    }
    printf("l %s\n",luma_matrix_file);
    printf("c %s\n",chroma_matrix_file);
    printf("q %s\n",qscale_file);
    printf("w %s\n",width);
    printf("h %s\n",height);
    printf("i %s\n",input_file);
    printf("o %s\n",output_file);
    printf("m %s\n",block_num);

    return SetLumaMatrix(luma_matrix_file);

}
#endif
/*
 * ./encoder [-l luma_matrix_file] [-c  chroma_matrix_file] [-q qscale_file] [-w width] [-h height] [-m block_num_of_macroblock] -i input_file -o output_file
 */
int main(int argc, char **argv)
{
#if 1
    int32_t ret = GetParam(argc, argv);
    return 0;
    if (ret < 0) {
        printf("error %d\n", __LINE__);
        return -1;
    }
#endif
    if (argc != 3) {
        printf("error %d\n", __LINE__);
        //return -1;
    }
    //FILE *input = fopen(argv[1], "r");
    FILE *input = fopen("./luma_matrix.txt", "r");
    if (input == NULL) {
        printf("err %s\n", argv[1]);
        return -1;
    }
    FILE *output = fopen(argv[2], "w");
    if (output == NULL) {
        printf("err %s\n", argv[2]);
        return -1;
    }
    //decode_init();
    uint32_t size = HORIZONTAL*VIRTICAL2* 2;
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
        uint8_t *frame = encode_frame(y_data, cb_data, cr_data, &frame_size);

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

