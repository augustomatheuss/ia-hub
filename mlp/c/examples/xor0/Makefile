CC=gcc
CFLAGS=-c -Wall -pedantic
MLP=../../mlp.c
LIBS=-lm

all: main

main: xor.o
	$(CC) $(MLP) $(LIBS) xor.o -o xor

xor.o: xor.c
	$(CC) xor.c $(CFLAGS)

clean:
	rm xor.o

