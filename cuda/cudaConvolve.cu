#include <stdio.h>
#include <stdlib.h>
#include "cudaConvolve.h"
#include "cudaFunc.h"


__global__ void convolveGray(uint8_t *src, uint8_t *dst, int width, int height) {
	__shared__ int save[256];

	int r, c, kr, kc;
	// float kernel[3][3] = {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}};
	float kernel[3][3] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
	
	size_t x = blockIdx.x*blockDim.x + threadIdx.x;
	size_t y = blockIdx.y*blockDim.y + threadIdx.y;

	//bounds check
	if (0 < x && x < height && 0 < y && y < width) {
		float total = 0;
		for (r = x-1, kr = 0 ; r <= x+1 ; r++, kr++){
			for (c = y-1, kc = 0 ; c <= y+1 ; c++, kc++){
				float sum = 0;
				int pixel = src[width * r + c]; 
				if (save[pixel]){
					sum = save[pixel];
				}
				else{
					sum = pixel  * kernel[kr][kc] / 1.0;
				}
				total += sum;
			}
		}
		dst[width * x + y] = total;
	}
}

__global__ void convolveRGB(uint8_t *src, uint8_t *dst, int width, int height) {
	int rr = 0;
	int c = 0;
	int kc = 0;
	int kr = 0;
	__shared__ int save[256];

	//float kernel[3][3] = {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}};
	// int kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
	float kernel[3][3] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};

	size_t x = blockIdx.x*blockDim.x + threadIdx.x;
	size_t y = blockIdx.y*blockDim.y + threadIdx.y;

	//bounds check
	if (0 < x && x < height && 0 < y && y < 3*width) {
		float r = 0, g = 0, b = 0;
		for (rr = x-1, kr = 0 ; rr <= x+1 ; rr++, kr++) {
			for (c = (y*3)-3, kc = 0 ; c <= (y*3)+3 ; c+=3, kc++) {
				float sumR = 0;
				float sumG = 0;
				float sumB = 0;

				int pixelR = src[(width*3) * rr + c];
				if (save[pixelR]){
					sumR = save[pixelR];
				}else{
					sumR = pixelR * kernel[kr][kc] /1.0;
				}
				r += sumR;

				int pixelG = src[(width*3) * rr + c+1];
				if (save[pixelG]){
					sumG = save[pixelG];
				}else{
					sumG = pixelG * kernel[kr][kc] /1.0;
				}
				g += sumG;

				int pixelB = src[(width*3) * rr + c+2];
				if (pixelB){
					sumB = save[pixelB];
				}
				else{
					sumB = pixelB* kernel[kr][kc] /1.0;
				}
				b += sumB;				 
			}
		}
		dst[width*3 * x + (y*3)] = r;
		dst[width*3 * x + (y*3)+1] = g;
		dst[width*3 * x + (y*3)+2] = b;
	}
}


extern "C" void gpuConvolve(uint8_t *src, int width, int height, int val, size_t bytes, int iter){
	uint8_t *gpuSrc; 
    cudaMalloc(&gpuSrc, bytes * sizeof(uint8_t));
    uint8_t *gpuDst;
    cudaMalloc(&gpuDst, bytes * sizeof(uint8_t));

    cudaMemcpy(gpuSrc, src, bytes, cudaMemcpyHostToDevice); 
	const int blockSize = 16; //multiple of warp

	int i = 0;
	for (i = 0; i < iter; i++){
		if (val == 1) {
			int gx = height/blockSize;
			if (height%blockSize != 0){
				gx++;
			} 
			int gy = width/blockSize;
			if (width%blockSize != 0){
				gy++;
			} 
			dim3 grid(gx, gy);
			dim3 block(blockSize, blockSize);
			convolveGray<<<grid, block>>>(gpuSrc, gpuDst, width, height);
		} else {
			int gx = height/blockSize;
			if (height%blockSize != 0){
				gx++;
			} 
			int gy = width*3/blockSize;
			if (width%blockSize != 0){
				gy++;
			}
			dim3 grid(gx, gy);
			dim3 block(blockSize, blockSize);
			convolveRGB<<<grid, block>>>(gpuSrc, gpuDst, width, height);
		}
	}
    cudaThreadSynchronize();
    cudaMemcpy(src, gpuDst, bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpuDst);
    cudaFree(gpuSrc);
}