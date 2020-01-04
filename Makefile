#
# Copyright (c) 2020 Yuusuke Miyazaki
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#
all:encoder


CFLAGS= -g -I./ -Wall

encoder:encoder.o qscale.o slice_size.o code_size.o dct.o
	gcc -o encoder ${CFLAGS} encoder.o qscale.o slice_size.o code_size.o dct.o -lm

encoder.o:encoder.c
	gcc ${CFLAGS} -c encoder.c -lm

qscale.o:qscale.c
	gcc ${CFLAGS} -c qscale.c

slice_size.o:slice_size.c
	gcc ${CFLAGS} -c slice_size.c

code_size.o:code_size.c
	gcc ${CFLAGS} -c code_size.c

dct.o:dct.c
	gcc ${CFLAGS} -c dct.c


clean:
	rm -f *.o encoder 
