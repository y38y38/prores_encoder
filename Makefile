#
# Copyright (c) 2020 Yuusuke Miyazaki
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#
all: .deps libproresencoder.so libproresencoder.a encoder Capture

CC=g++

SDK_PATH=../bmd_capture/include
CFLAGS=  -g  -I./ -Wall -Wno-multichar -I $(SDK_PATH)  -fno-rtti
SRCS = frame.c dct.c bitstream.c vlc.c debug.c slice.c 
OBJS = $(SRCS:..c=.o)
LDFLAGS=-L $(ENCODER_PATH) -lm -ldl -lpthread 

.deps:
	$(CC) -M ${CFLAGS} $(SRCS) main.c -fPIC > $@

libproresencoder.a: $(OBJS)
	ar rv $@ $^
	ranlib $@

libproresencoder.so: $(OBJS)
	$(CC) -shared -o $@.1.0 $^ -fPIC


encoder:main.c libproresencoder.so.1.0
	${CC} ${CFLAGS}  -o $@ $^ -lm -lpthread


Capture: Capture.cpp Config.cpp $(SDK_PATH)/DeckLinkAPIDispatch.cpp
	$(CC) -o Capture Capture.cpp Config.cpp libproresencoder.so.1.0 $(SDK_PATH)/DeckLinkAPIDispatch.cpp $(CFLAGS) $(LDFLAGS)



clean:
	rm -f *.o libproresencoder.so.1.0 libproresencoder.a encoder

install:
	cp libproresencoder.so.1.0 libproresencoder.a /usr/lib/