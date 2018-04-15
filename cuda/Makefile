cudaConv: main.o cudaFunc.o cudaConvolve.o
	nvcc -o cudaConv main.o cudaFunc.o cudaConvolve.o -O3

cudaConvolve.o: cudaConvolve.cu
	nvcc -c cudaConvolve.cu

main.o: main.c 
	gcc -c main.c -O3

cudaFunc.o: cudaFunc.c
	gcc -c cudaFunc.c -O3

clean:
	\rm -f *.o cudaConv