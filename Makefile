#
# Copyright (c) 2020 Yuusuke Miyazaki
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#
all:encoder

cuda:encoder_cuda

CC=nvcc


ifeq ($(MAKECMDGOALS),cuda)
	CFLAGS=  -g -O3 -DCUDA_ENCODER -I./ 
else
	CFLAGS=  -O3 -I./ 
endif


encoder_cuda:frame.o main.o bitstream.o slice.o debug.o vlc.o dct.o dct_init.o
	${CC} -o encoder ${CFLAGS} -pg frame.o main.o  bitstream.o  slice.o  debug.o  vlc.o dct.o dct_init.o -lm -lpthread

encoder:frame.o main.o bitstream.o slice.o debug.o vlc.o dct.o dct_init.o
	${CC} -o encoder ${CFLAGS} -pg frame.o main.o  bitstream.o  slice.o  debug.o  vlc.o dct.o dct_init.o -lm -lpthread

vlc.o:vlc.cu
	${CC} ${CFLAGS} -c vlc.cu

dct.o:dct.cu
	${CC} ${CFLAGS}  -c dct.cu

dct_init.o:dct_init.cu
	${CC} ${CFLAGS}  -c dct_init.cu


frame.o:frame.cu
	${CC} ${CFLAGS}  -c frame.cu -lm


bitstream.o:bitstream.cu
	${CC} ${CFLAGS}  -c bitstream.cu

bitstream_cuda.o:bitstream_cuda.cu
	${CC} ${CFLAGS} -c bitstream_cuda.cu


debug.o:debug.cu
	${CC} ${CFLAGS}  -c debug.cu


slice.o:slice.cu
	${CC} ${CFLAGS}  -c slice.cu

main.o:main.cu
	${CC} ${CFLAGS}  -c main.cu


clean:
	rm -f *.o encoder 
