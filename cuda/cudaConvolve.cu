#include <stdio.h>
#include <stdlib.h>
#include "cudaConvolve.h"
#include "cudaFunc.h"


__global__ void convolveGray(uint8_t *src, uint8_t *dst, int width, int height) {
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
				total += src[width * r + c] * kernel[kr][kc] / 1.0;
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
				r += src[(width*3) * rr + c]* kernel[kr][kc] /1.0;
				g += src[(width*3) * rr + c+1] * kernel[kr][kc] /1.0;
				b += src[(width*3) * rr + c+2] * kernel[kr][kc] /1.0;
			}
		}
		dst[width*3 * x + (y*3)] = r;
		dst[width*3 * x + (y*3)+1] = g;
		dst[width*3 * x + (y*3)+2] = b;
	}
}


extern "C" void gpuConvolve(uint8_t *src, int width, int height, int val, size_t bytes){
	uint8_t *gpuSrc; 
    cudaMalloc(&gpuSrc, bytes * sizeof(uint8_t));
    uint8_t *gpuDst;
    cudaMalloc(&gpuDst, bytes * sizeof(uint8_t));

    cudaMemcpy(gpuSrc, src, bytes, cudaMemcpyHostToDevice); 
	const int blockSize = 32; //multiple of warp

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

    cudaThreadSynchronize();
    cudaMemcpy(src, gpuDst, bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpuDst);
    cudaFree(gpuSrc);
}