#include <stdio.h>
#include <stdlib.h>
#include "cudaConv.h"

__global__ void padKernel_kernel(int kernelW, int kernelH, int fftH, int fftW, double *src, double *dst,int kernelX,int kernelY){
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    int diffx = 0;
    int diffy = 0;
    if (y < kernelH && x < kernelW){
        diffy = y - kernelY;
        diffx = x - kernelX;
        if (diffy < 0){
            diffy += fftH;
        }
        if (diffx < 0){
            diffx += fftW;
        }
        dst[diffy * fftW + diffx] = src[y * kernelW + x];
    }
}

extern "C" void padKernel(int kernelW, int kernelH, int fftH, int fftW, double *src, double *dst,int kernelX,int kernelY){
    dim3 threads(32, 8);
    int a = 1+ kernelW/threads.x;
    if (kernelW%threads.x == 0){
    	a = kernelW/threads.x;
    }
    int b = 1+ kernelH/threads.y;
    if (kernelH%threads.y == 0){
    	b = kernelH/threads.y;
    }
    dim3 grid(a, b);

    padKernel_kernel<<<grid, threads>>>(kernelW,kernelH,fftH,fftW,src,dst,kernelX,kernelY);
}

__global__ void padData_kernel(int fftW, int fftH, int width, int height, int kernelX, int kernelY, float *src, float *dst){
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int h = height + kernelY;
    int w = width + kernelX;

    int diffy = 0;
    int diffx = 0;
    if (y < fftH && x < fftW){
        if (y < height){
            diffy = y;
        }
        if (x < width){
            diffx = x;
        }
        if (y >= height && y < h){
            diffy = height - 1;
        }
        if (x >= width && x < w){
            diffx = width - 1;
        }
        if (y >= h){
            diffy = 0;
        }
        if (x >= w){
            diffx = 0;
        }

        dst[y * fftW + x] = src[diffy * width + diffx];
    }
}

extern "C" void padData(int fftW, int fftH, int width, int height, int kernelX, int kernelY, float *src,float *dst){
    dim3 threads(32, 8);
       int a = 1+ fftW/threads.x;
    if (fftW%threads.x == 0){
    	a = fftW/threads.x;
    }
    int b = 1+ fftH/threads.y;
    if (fftH%threads.y == 0){
    	b = fftH/threads.y;
    }
    dim3 grid(a, b);

    padData_kernel<<<grid, threads>>>(fftW,fftH,width,height,kernelX,kernelY,src,dst);
}