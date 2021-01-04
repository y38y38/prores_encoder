#
# Copyright (c) 2020 Yuusuke Miyazaki
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#
all:encoder

cuda:encoder_cuda

CC=nvcc

CFLAGS=  -g -O2 -I./ 

encoder:frame.o main.o bitstream.o slice.o debug.o
	${CC} -o encoder ${CFLAGS} frame.o main.o  bitstream.o  slice.o  debug.o -lm -lpthread

#encoder_cuda:frame.o main.o bitstream.o slice.o debug.o vlc.o dct.o
#	${CC} -o encoder ${CFLAGS} frame.o main.o  bitstream.o  slice.o  debug.o  vlc.o dct.o -lm -lpthread

vlc.o:vlc.cu
	${CC} ${CFLAGS} -c vlc.cu

dct.o:dct.cu
	${CC} ${CFLAGS} -c dct.cu

frame.o:frame.cu
	${CC} ${CFLAGS} -c frame.cu -lm


bitstream.o:bitstream.cu
	${CC} ${CFLAGS} -c bitstream.cu


debug.o:debug.cu
	${CC} ${CFLAGS} -c debug.cu


slice.o:slice.cu
	${CC} ${CFLAGS} -c slice.cu

main.o:main.cu
	${CC} ${CFLAGS} -c main.cu


clean:
	rm -f *.o encoder 
