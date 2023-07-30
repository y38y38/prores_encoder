#
# Copyright (c) 2020 Yuusuke Miyazaki
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#
all: .deps libproresencoder.so libproresencoder.a encoder

CC=gcc

CFLAGS=  -g  -I./ -Wall
SRCS = frame.c dct.c bitstream.c vlc.c debug.c slice.c 
OBJS = $(SRCS:..c=.o)

.deps:
	$(CC) -M ${CFLAGS} $(SRCS) main.c > $@

libproresencoder.a: $(OBJS)
	ar rv $@ $?
	ranlib $@

libproresencoder.so: $(OBJS)
	$(CC) -shared -o $@.1.0 $^ -fPIC


encoder:main.c libproresencoder.so.1.0
	${CC} ${CFLAGS}  -o $@ $^ -lm -lpthread



clean:
	rm -f *.o libproresencoder.so libproresencoder.a encoder
