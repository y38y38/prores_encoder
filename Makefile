#
# Copyright (c) 2020 Yuusuke Miyazaki
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#
all:encoder

CC=nvcc

CFLAGS=  -g -O2 -I./

encoder:frame.o   dct.o main.o bitstream.o slice.o vlc.o debug.o
	${CC} -o encoder ${CFLAGS} frame.o  dct.o main.o  bitstream.o  vlc.o slice.o  debug.o -lm -lpthread

frame.o:frame.cu
	${CC} ${CFLAGS} -c frame.cu -lm


dct.o:dct.cu
	${CC} ${CFLAGS} -c dct.cu

bitstream.o:bitstream.cu
	${CC} ${CFLAGS} -c bitstream.cu

vlc.o:vlc.cu
	${CC} ${CFLAGS} -c vlc.cu

debug.o:debug.cu
	${CC} ${CFLAGS} -c debug.cu


slice.o:slice.cu
	${CC} ${CFLAGS} -c slice.cu

main.o:main.cu
	${CC} ${CFLAGS} -c main.cu


clean:
	rm -f *.o encoder 
