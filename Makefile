#
# Copyright (c) 2020 Yuusuke Miyazaki
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#
all:encoder

CC=gcc

CFLAGS=  -g -O2 -I./

encoder:frame.o   dct.o main.o bitstream.o slice.o vlc.o debug.o
	${CC} -o encoder ${CFLAGS} frame.o  dct.o main.o  bitstream.o  vlc.o slice.o  debug.o -lm -lpthread

frame.o:frame.c
	${CC} ${CFLAGS} -c frame.c -lm


dct.o:dct.c
	${CC} ${CFLAGS} -c dct.c

bitstream.o:bitstream.c
	${CC} ${CFLAGS} -c bitstream.c

vlc.o:vlc.c
	${CC} ${CFLAGS} -c vlc.c

debug.o:debug.c
	${CC} ${CFLAGS} -c debug.c


slice.o:slice.c
	${CC} ${CFLAGS} -c slice.c

main.o:main.c
	${CC} ${CFLAGS} -c main.c


clean:
	rm -f *.o encoder 
