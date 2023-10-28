#
# Copyright (c) 2020 Yuusuke Miyazaki
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#
all: make_dir $(OBJS) libproresencoder.so libproresencoder.a

CC=gcc
BIN_DIR= ./bin/

CFLAGS=  -g  -I./ -Wall -fPIC
SRCS = frame.c dct.c bitstream.c vlc.c debug.c slice.c 
OBJS = $(SRCS:.c=.o)
LDFLAGS= -lm -ldl -lpthread 

make_dir:
	mkdir -p ${BIN_DIR}


#encoder:main.c ${BIN_DIR}/libproresencoder.so.1.0
#	${CC} ${CFLAGS}  -o ${BIN_DIR}/$@ $^ ${LDFLAGS}

libproresencoder.a: $(OBJS)
	ar rv ${BIN_DIR}/$@ $^
	ranlib ${BIN_DIR}/$@

libproresencoder.so: $(OBJS)
	$(CC) -shared -o ${BIN_DIR}/$@.1.0 $^ -fPIC



$(OBJS): $(SRCS)
	$(CC) $(CFLAGS) -c $(SRCS)

clean:
	rm -rf *.o ${BIN_DIR}

install:
	cp ${BIN_DIR}/libproresencoder.so.1.0 ${BIN_DIR}/libproresencoder.a /usr/lib/