conv: main.o  
	gcc -o conv main.o -fopenmp -O3 -lfftw3 -lm

main.o: main.c 
	gcc -c main.c -L  -03

clean:
	\rm -f *.o conv