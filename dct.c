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
#include <string.h>
#include <math.h>

#define MAX_X   (8)
#define MAX_Y   (8)

#define KEI     (1)

static double dct(int16_t *block, int h, int v)
{
    int x=0,y=0;
    double value = 0;
    for(y=0;y<MAX_Y;y++) {
        for(x=0;x<MAX_X;x++) {
            double kc = cos((M_PI * v * ((2.0 * y) + 1.0)) / 16.0) * cos((M_PI * h * ((2.0 * x) + 1.0)) / 16.0);
            //printf("kc %f %d %d\n", kc, h, v);
            value += block[(y * 8) + x] *  kc;
        }
    }
    //printf(" %f \n", value);
    if ((h==1) && (v==0)) {
        //printf("\n h1 v0 %lf\n", value);
    }
    if ( h == 0) {
        value *= KEI/ sqrt(2.0);
    } else {
        value *= KEI;
    }
    //printf(" %f \n", value);
    if (v == 0) {
        value *= KEI / sqrt(2.0);
    } else {
        value *= KEI;
    }
    //printf(" %f \n", value);

    value = value / 4;
    //printf(": %f \n", value);
    //return value ;
    return value;
}
void print_block(int16_t *block)
{

    int x,y;
    for (y=0;y<MAX_Y;y++) {
        for (x=0;x<MAX_X;x++) {
            printf("%d ", block[(y * MAX_X) + x]);
        }
        printf("\n");
    }
    printf("\n");
}
double idct(int16_t *block, int x, int y)
{
    int h=0,v=0;
    double value = 0;
    double c_u=0;
    double c_v=0;
    for(h=0;h<MAX_Y;h++) {
        for(v=0;v<MAX_X;v++) {
            if (h==0) {
                c_u = KEI / sqrt(2.0);
            } else {
                c_u = KEI;
            }
            if (v==0) {
                c_v = KEI /sqrt(2.0);
            } else {
                c_v = KEI;
            }
            value += c_v * c_u * block[(v * MAX_X) + h] * cos((M_PI * v * ( (2.0 * y) + 1.0)) / 16.0) * cos((M_PI * h  * ((2.0 * x) + 1.0)) / 16.0);
        }
    }
    return value / 4;
}





int g_first = 1;
int dct_block(int16_t *block) {
#if 0
    printf("before\n");
    print_block(block);
    if (g_first == 0) {
        printf("orginal\n");
        print_block(block);
    }
#endif
    int h,v,i;
    double result[MAX_X * MAX_Y];
    for (v=0;v<MAX_Y;v++) {
        for (h=0;h<MAX_X;h++) {
            result[(v * MAX_X) + h] = dct(block, h, v);
        }
    }
    for(i = 0;i<MAX_X*MAX_Y;i++) {
        block[i] = (int16_t)result[i];
    }
    if (g_first == 0) {
        printf("after\n");
        print_block(block);
    }
#if 0
    printf("after\n");
    print_block(block);
#endif
    g_first++;

    return 0;
}
int idct_block(int16_t *block) {

    int h,v,i;
    double result[MAX_X * MAX_Y];
    for (v=0;v<MAX_Y;v++) {
        for (h=0;h<MAX_X;h++) {
            result[(v * MAX_X) + h] = idct(block, h, v);
        }
    }
    for(i = 0;i<MAX_X*MAX_Y;i++) {
        block[i] = (int16_t)result[i];
    }

    return 0;
}
#if 0
#if 0
int16_t org[MAX_X * MAX_Y] =
{
-172, -172, -172, -173, -175, -170, -158, -131, 
-171, -172, -173, -173, -170, -159, -137, -117, 
-172, -172, -171, -166, -154, -136, -117, -103, 
-172, -170, -164, -152, -133, -115, -98 , -94 ,
-170, -165, -153, -136, -113, -96 , -87 , -96 ,
-160, -150, -139, -122, -103, -93 , -91 , -104, 
-145, -134, -125, -115, -107, -102, -104, -114, 
-130, -119, -113, -111, -112, -114, -118, -125, 
};
#else
int16_t org[MAX_X * MAX_Y] =
{
698, 667, 616, 640, 666, 639, 642, 657, 
659, 656, 656, 657, 656, 657, 655, 649,
655, 654, 652, 649, 654, 668, 660, 647,
641, 568, 551, 641, 689, 691, 669, 657, 
654, 641, 623, 597, 588, 595, 629, 656,
653, 666, 669, 667, 672, 683, 682, 681, 
693, 695, 680, 669, 664, 664, 645, 641, 
646, 660, 665, 662, 664, 665, 663, 663, 
};

#endif

int main(void) {
    int x,y,h,v;
    printf("orginal\n");
    for (y=0;y<MAX_Y;y++) {
        for (x=0;x<MAX_X;x++) {
            printf("%d ", org[(y * MAX_X) + x]);
        }
        printf("\n");
    }
    printf("\n");

    printf("dct result\n");
    dct_block(org);

    for (v=0;v<MAX_Y;v++) {
        for (h=0;h<MAX_X;h++) {
            printf("%d ", org[(v * MAX_X) + h]);
        }
        printf("\n");
    }
    printf("idct result\n");
    idct_block(org);
    for (v=0;v<MAX_Y;v++) {
        for (h=0;h<MAX_X;h++) {
            printf("%d ", org[(v * MAX_X) + h]);
        }
        printf("\n");
    }

    printf("\n");
    return 0;
}
#endif
