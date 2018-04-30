#include <stdio.h>
#include <stdlib.h>
#include "cudaConvolve.h"
#include "cudaFunc.h"


__global__ void convolveGray(uint8_t *src, uint8_t *dst, uint8_t *tmp,int width, int height) {
	__shared__ int save[17][17];
	__shared__ int save1[17][17];

	int r, c, kr, kc;
	float row_k[3]={1,2,1};
	float col_k[3]={-1,0,1};
	
	size_t x = blockIdx.x*blockDim.x + threadIdx.x;
	size_t y = blockIdx.y*blockDim.y + threadIdx.y;
	//row convolution filter

	if (0 < x && x < height && 0 < y && y < width) {
		save[threadIdx.x][threadIdx.y]= src[x*width+y];
		save1[threadIdx.x][threadIdx.y]= tmp[x*width+y];
	}

	__syncthreads();

	int pixel = 0;
	int i = 0;
	int j = 0;
	//bounds check
	if (0 < x && x < height && 0 < y && y < width) {
		float total = 0;

		r = x-1;
		kr = 0;
		i = r-(blockIdx.x*blockDim.x);
		pixel = save[i][threadIdx.y];
		//pixel = src[width * r+y]; 
		total += pixel  * row_k[kr] / 1.0;

		r = x;
		kr = 1;
		i = r-(blockIdx.x*blockDim.x);
		pixel = save[i][threadIdx.y];
		// pixel = src[width * r+y]; 
		total += pixel  * row_k[kr] / 1.0;

		r = x+1;
		kr = 2;
		i = r-(blockIdx.x*blockDim.x);
		pixel = save[i][threadIdx.y];
		// pixel = src[width * r+y]; 
		total += pixel  * row_k[kr] / 1.0;
		tmp[width * x + y] = total;
	}
	//column convolution filter
	if (0 < x && x < height && 0 < y && y < width) {
		float total = 0;
		c = y-1;
		kc = 0;
		j = c-(blockIdx.y*blockDim.y);
		pixel = save1[threadIdx.x][c];
		// pixel = tmp[width * x + c]; 
		total += pixel  * col_k[kc] / 1.0;

		c = y;
		kc = 1;
		j = c-(blockIdx.y*blockDim.y);
		pixel = save1[threadIdx.x][c];
		// pixel = tmp[width * x + c]; 
		total += pixel  * col_k[kc] / 1.0;

		c = y+1;
		kc = 2;
		j = c-(blockIdx.y*blockDim.y);
		pixel = save1[threadIdx.x][c];
		// pixel = tmp[width * x + c]; 
		total += pixel  * col_k[kc] / 1.0;
	
		dst[width * x + y] = total;
	}
}

__global__ void convolveRGB(uint8_t *src, uint8_t *dst, uint8_t *tmp, int width, int height) {
	int rr = 0;
	int c = 0;
	int kc = 0;
	int kr = 0;
	__shared__ uint8_t save[32][32*3];
	__shared__ uint8_t save1[32][32*3];

	float row_k[3]={1,2,1};
	float row_c[3]={-1,0,1};
	size_t x = blockIdx.x*blockDim.x + threadIdx.x;
	size_t y = blockIdx.y*blockDim.y + threadIdx.y;

	if (0 < x && x < height && 0 < y && y < 3*width) {
		save[threadIdx.x][threadIdx.y]= src[(x*width*3)+y];
		save1[threadIdx.x][threadIdx.y]= tmp[(x*width*3)+y];
	}

	__syncthreads();

	int pixelR = 0;
	int pixelG = 0;
	int pixelB = 0;
	rr = x-1;
	kr = 0;
	int i = 0;
	int j = 0;
	//bounds check
	if (0 < x && x < height && 0 < y && y < 3*width) {
		float r = 0, g = 0, b = 0;
		rr = x-1;
		kr = 0;
		i = rr-(blockIdx.x*blockDim.x);
		pixelR = save[i][(1*y)];
		pixelG = save[i][(1*y)+1];
		pixelB = save[i][(1*y)+2];

		r += pixelR * row_k[kr] /1.0;
		g += pixelG * row_k[kr]/1.0;
		b += pixelB* row_k[kr]/1.0;	

		rr = x;
		kr = 1;
		i = rr-(blockIdx.x*blockDim.x);
		pixelR = save[i][(1*y)];
		pixelG = save[i][(1*y)+1];
		pixelB = save[i][(1*y)+2];

		r += pixelR * row_k[kr] /1.0;
		g += pixelG * row_k[kr]/1.0;
		b += pixelB* row_k[kr]/1.0;	

		rr = x+1;
		kr = 2;
		i = rr-(blockIdx.x*blockDim.x);
		pixelR = save[i][(1*y)];
		pixelG = save[i][(1*y)+1];
		pixelB = save[i][(1*y)+2];

		r += pixelR * row_k[kr] /1.0;
		g += pixelG * row_k[kr]/1.0;
		b += pixelB* row_k[kr]/1.0;		

		
		tmp[width*3 * x + (y*3)] = r;
		tmp[width*3 * x + (y*3)+1] = g;
		tmp[width*3 * x + (y*3)+2] = b;
	}
	if (0 < x && x < height && 0 < y && y < 3*width) {
		float r = 0, g = 0, b = 0;
		c = (y*3)-3;
		kc = 0;
		j = c-(blockIdx.y*blockDim.y);
		pixelR = save1[threadIdx.x][c];
		pixelG = save1[threadIdx.x][c+1];
		pixelB = save1[threadIdx.x][c+2];
		r += pixelR * row_c[kc] /1.0;
		g += pixelG * row_c[kc]/1.0;
		b += pixelB* row_c[kc]/1.0;		

		c = (y*3);
		kc = 1;
		j = c-(blockIdx.y*blockDim.y);
		pixelR = save1[threadIdx.x][c];
		pixelG = save1[threadIdx.x][c+1];
		pixelB = save1[threadIdx.x][c+2];
		r += pixelR * row_c[kc] /1.0;
		g += pixelG * row_c[kc]/1.0;
		b += pixelB* row_c[kc]/1.0;	

		c = (y*3)+3;
		kc = 2;
		j = c-(blockIdx.y*blockDim.y);
		pixelR = save1[threadIdx.x][c];
		pixelG = save1[threadIdx.x][c+1];
		pixelB = save1[threadIdx.x][c+2];
		r += pixelR * row_c[kc] /1.0;
		g += pixelG * row_c[kc]/1.0;
		b += pixelB* row_c[kc]/1.0;	

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
	uint8_t *gpuTmp;
    cudaMalloc(&gpuTmp, bytes * sizeof(uint8_t));
    
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
			convolveGray<<<grid, block>>>(gpuSrc, gpuDst,gpuTmp, width, height);
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
			convolveRGB<<<grid, block>>>(gpuSrc, gpuDst, gpuTmp,width, height);
		}
	}
    cudaThreadSynchronize();
    cudaMemcpy(src, gpuDst, bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpuDst);
    cudaFree(gpuSrc);
    cudaFree(gpuTmp);
}
