#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "encoder.h"
int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("error %d\n", __LINE__);
        return -1;
    }
    FILE *input = fopen(argv[1], "r");
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

