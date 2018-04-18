conv: main.o serial.o 
	nvcc -o conv main.o serial.o -O3

main.o: main.c 
	gcc -c main.c -O3

serial.o: serial.c
	gcc -c serial.c -O3

clean:
	\rm -f *.o conv