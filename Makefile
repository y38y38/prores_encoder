#
# Copyright (c) 2020 Yuusuke Miyazaki
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#
all:encoder


CFLAGS= -g -I./ -Wall

encoder:frame.o  slice_size.o  dct.o main.o bitstream.o slice.o
	gcc -o encoder ${CFLAGS} frame.o  slice_size.o  dct.o main.o  bitstream.o  slice.o -lm

frame.o:frame.c
	gcc ${CFLAGS} -c frame.c -lm

slice_size.o:slice_size.c
	gcc ${CFLAGS} -c slice_size.c


dct.o:dct.c
	gcc ${CFLAGS} -c dct.c

bitstream.o:bitstream.c
	gcc ${CFLAGS} -c bitstream.c

slice.o:slice.c
	gcc ${CFLAGS} -c slice.c

main.o:main.c
	gcc ${CFLAGS} -c main.c


clean:
	rm -f *.o encoder 
