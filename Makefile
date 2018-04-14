cuda_conv: main.o serial.o 
	nvcc -o conv main.o serial.o -O2

main.o: main.c 
	gcc -c main.c -O2

funcs.o: funcs.c
	gcc -c serial.c -O2

clean:
	\rm -f *.o cuda_conv