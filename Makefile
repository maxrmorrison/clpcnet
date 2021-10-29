files = src/celt_lpc.c src/freq.c src/kiss_fft.c src/lpcnet_enc.c src/pitch.c src/preprocess.c

all: preprocess

preprocess:
	mkdir -p bin/
	gcc -Wall -W -O3 -g -I src/ $(files) -o bin/preprocess -lm

clean:
	rm -rf bin/preprocess
