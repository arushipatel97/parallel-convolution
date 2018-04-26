all: mpiConv

mpiConv : main.o serial.o 
	mpicc main.o serial.o -o mpiConv -lm

main.o: main.c 
	mpicc -c main.c

serial.o: serial.c
	gcc -c serial.c -O3

clean:
	\rm -f *.o mpiConv