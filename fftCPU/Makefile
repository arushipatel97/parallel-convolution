conv: main.o serial.o 
	nvcc -o conv main.o serial.o -O3 -lfftw3 -lm

main.o: main.c 
	gcc -c main.c -L -03

serial.o: serial.c
	gcc -c serial.c -O3

clean:
	\rm -f *.o conv