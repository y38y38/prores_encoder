#
# Copyright (c) 2020 Yuusuke Miyazaki
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#
all:encoder


CFLAGS=  -g -I./ -Wall

encoder:frame.o   dct.o main.o bitstream.o slice.o vlc.o debug.o
	gcc -o encoder ${CFLAGS} frame.o  dct.o main.o  bitstream.o  vlc.o slice.o  debug.o -lm -lpthread

frame.o:frame.c
	gcc ${CFLAGS} -c frame.c -lm


dct.o:dct.c
	gcc ${CFLAGS} -c dct.c

bitstream.o:bitstream.c
	gcc ${CFLAGS} -c bitstream.c

vlc.o:vlc.c
	gcc ${CFLAGS} -c vlc.c

debug.o:debug.c
	gcc ${CFLAGS} -c debug.c


slice.o:slice.c
	gcc ${CFLAGS} -c slice.c

main.o:main.c
	gcc ${CFLAGS} -c main.c


clean:
	rm -f *.o encoder 
