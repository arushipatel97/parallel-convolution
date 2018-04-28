#include <stdio.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "serial.h"
#include <sys/time.h>
// #include <fftw.h>
#include "fftw3.h"

void padKernel(int kernelW, int kernelH, int fftH, int fftW, double* src, double* dst, int kernelX, int kernelY){
	int y = 0;
	int x = 0;
	int diffy = 0;
	int diffx = 0;
	for (y = 0; y < kernelH; y++){
		for (x = 0; x < kernelW; x++){
			diffy = y - kernelY;
			diffx = x - kernelX;
			if (diffy < 0){
				diffy+= fftH;
			}
			if (diffx < 0){
				diffx += fftW;
			}
			dst[(diffy*fftW)+diffx] = (double)src[(y*kernelW)+x];
		}
	}
}

void padData(int fftW, int fftH, int width, int height, int kernelH, int kernelW, int kernelX, int kernelY, uint8_t* src, uint8_t* dst){
	int x = 0;
	int y = 0;
	int h = height+kernelY;
	int w = width +kernelW;
	int diffy = 0;
	int diffx = 0;
	for (y = 0; y < kernelH; y++){
		for (x = 0; x < kernelW; x++){
			if (y < height){
				diffy = y;
			}
			if (x < width){
				diffx = x;
			}
			if (y>=height && y < h){
				diffy = h-1;
			}
			if (x>= width && x < w){
				diffx = w-1;
			}
			if (y>= h){
				diffy = 0;
			}
			if (x >=w){
				diffx = 0;
			}
			dst[(y*fftW)+x] = src[(diffy*width)+diffx]; 
		}
	}
}

void mult(fftw_complex* fftKernelFT, fftw_complex* fftDst, fftw_complex* output, int fftW, int fftH){
	int x = 0;
	int y = 0;
	for (y = 0; y < fftH; y++){
		for (x = 0; x < fftW; x++){
			// fftw_complex a = fftKernelFT[x*fftW+y];
			fftw_complex b = 
			// 
			{(fftKernelFT[x*fftW+y][0]*fftDst[x*fftW+y][0]-fftKernelFT[x*fftW+y][1]*fftDst[x*fftW+y][1]),
				(fftKernelFT[x*fftW+y][1]* fftDst[x*fftW+y][0]+fftKernelFT[x*fftW+y][0]*fftDst[x*fftW+y][1])};
			memcpy(output+((x*fftW)+y), b, sizeof(fftw_complex));
			// output[x*fftW+y] = b;
		}
	}
}

int main(int argc, char** argv) {
	/* Count time */ 
	uint64_t c = micro_time(); 
	int fd, width, height, val, iter;

	int bpp;
	setParams(argc, argv, &width, &height, &val, &iter);	

    uint8_t* src = stbi_load(argv[1], &width, &height, &bpp, val);
	fprintf(stderr, "H:%d w:%d\n", height, width);
	
	uint8_t* dst= malloc(width*height*val);
	
	const int kernelW = 3;
	const int kernelH = 3;

	const int kernelX = 0;
	const int kernelY = 0;

	//size up
	const int fftH = roundtoFFt(height + kernelH - 1);
    const int fftW = roundtoFFt(width + kernelW - 1);
    fprintf(stderr, "ffth:%d fftw:%d\n", fftH, fftW);
    // fftw_plan forward;
    // forward=fftw2d_create_plan(fftW,fftH, FFTW_REAL_TO_COMPLEX, 0);

    fftw_complex* kernel = fftw_malloc(kernelW*kernelH*sizeof(fftw_complex));
    memset(kernel, 0, sizeof(fftw_complex)*kernelW*kernelH);
    memset(kernel+4, 1, sizeof(fftw_complex));

    fftw_complex* fftKernel = fftw_malloc(fftW*fftH*sizeof(fftw_complex));
    memset(fftKernel, 0, sizeof(fftw_complex)*fftH*fftW);

    fftw_complex* fftKernelFT = fftw_malloc(fftW*fftH*sizeof(fftw_complex));
    memset(fftKernelFT, 0, sizeof(fftw_complex)*fftH*fftW);

    fftw_complex* fftSrc =fftw_malloc(fftW*fftH*sizeof(fftw_complex));
	fftw_complex* fftDst =fftw_malloc(fftW*fftH*sizeof(fftw_complex));
	memset(fftSrc, 0, sizeof(fftw_complex)*fftH*fftW);
	memset(fftDst, 0, sizeof(fftw_complex)*fftH*fftW);

	padKernel(kernelW, kernelH, fftH, fftW, (double*)kernel, (double*)fftKernel, kernelX, kernelY);
    padData(fftW, fftH, width, height, kernelH, kernelW, kernelX, kernelY, (uint8_t*) src, (uint8_t*)fftSrc);

	fftw_plan forwardK;	
    forwardK = fftw_plan_dft_2d(fftH, fftW, fftKernel, fftKernelFT, -1, 0);

    fftw_plan forwardD;	
    forwardD = fftw_plan_dft_2d(fftH, fftW, fftSrc, fftDst, -1, 0);

    fftw_execute(forwardK);
    fftw_execute(forwardD);

    fftw_complex* multOut = fftw_malloc(fftW*fftH*sizeof(fftw_complex));
    memset(multOut, 0, sizeof(fftw_complex)*fftH*fftW);

    mult(fftKernelFT, fftDst, multOut, fftW, fftH);


    //pad kernel
    //pad image
    //fft
    //fft
    //elementwise mult
    //ifft

	stbi_write_png("image_filter.png", width, height, val, dst, width*val);
	stbi_image_free(src);
	stbi_image_free(dst);
		/* compute time */
	c = micro_time() - c;
	double million = 1000 * 1000;
	fprintf(stdout, "Execution time: %.3f sec\n", c / million);

	return 0;
}

int roundtoFFt(int dataSize){
    int index;
    unsigned int low, hi;

    dataSize = roundUp(dataSize, 16);

    for (index = 31; index >= 0; index--){
        if (dataSize & ((uint32_t) 1 << index)){break;}
    }
    low = (uint32_t) 1 << index;
    if (low == (unsigned int)dataSize){
        return dataSize;
    }
    hi = (uint32_t) 1  << (index + 1);
    if (hi <= 1024){return hi;}
    else{
        return roundUp(dataSize, 512);
    }
}

int roundUp(int n, int m)
{
    int mod = n % m;

    if (!mod){
    	return m;
    }
    return n + m - mod;
}