#include <stdio.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <sys/time.h>
// #include <fftw.h>
#include "fftw3.h"
#include "cudaConv.h"
#include <stdlib.h>
#include <assert.h>
#include </usr/local/depot/cuda-8.0/include/cufft.h>
#include </usr/local/depot/cuda-8.0/include/cuda_runtime.h>


void setParams(int argc, char **argv, int *width, int *height, int* val, int* iter) {
	if (argc == 6 && !strcmp(argv[4], "gray")) {
		*width = atoi(argv[2]);
		*height = atoi(argv[3]);
		*val = 1;
		*iter = atoi(argv[5]);
	} else if (argc == 6 && !strcmp(argv[4], "rgb")) {
		*width = atoi(argv[2]);
		*height = atoi(argv[3]);
		*val = 3;
		*iter = atoi(argv[5]);
	} else {
		fprintf(stderr, "Incorrect Input!\n");
	}
}

uint64_t micro_time(void) {
	struct timeval tv;
	assert(gettimeofday(&tv, NULL) == 0);
	return tv.tv_sec * 1000 * 1000 + tv.tv_usec;
}


void mult(fftw_complex* fftKernelFT, fftw_complex* fftDst, fftw_complex* output, int fftW, int fftH){
	int x = 0;
	int y = 0;
	float c=1.0f/((float)(fftW*fftH));
	//float c =1.0f;
	int i;
	#pragma omp parallel for
	for (i = 0; i < fftW*fftH ; i++){
		fftw_complex b = 
		{c*(((fftKernelFT[i][0])*fftDst[i][0]-fftKernelFT[i][1]*fftDst[i][1])),
			c*((fftKernelFT[i][1]* fftDst[i][0]+fftKernelFT[i][0]*fftDst[i][1]))};
		output[i][0] = b[0];
		output[i][1] = b[1];
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
	

	const int kernelW = 3;
	const int kernelH = 3;

	const int kernelX = 0;
	const int kernelY = 0;

	//size up
	const int fftH = roundtoFFt(height + kernelH - 1);
    const int fftW = roundtoFFt(width + kernelW - 1);
    
    double* dSrc, *dPadData,*dKernel,*dPadKern;

    fftw_complex*dDS,*dKS;

    cufftHandle forward, inverse;

    double* kernel = malloc(kernelW*kernelH*sizeof(double));
    int i;
    for(i=0;i<kernelW*kernelH;i++){
     	kernel[i]=0.1111111111111111f;
    }

	cudaMalloc((void **)&dSrc, height * width * sizeof(float));
   	cudaMalloc((void **)&dKernel, kernelH * kernelW * sizeof(float));

    cudaMalloc((void **)&dPadData, fftH * fftW * sizeof(float));
    cudaMalloc((void **)&dPadKern, fftH * fftW * sizeof(float));

    cudaMalloc((void **)&dDS, fftH * (fftW / 2 + 1) * sizeof(fftw_complex));
    cudaMalloc((void **)&dKS, fftH * (fftW / 2 + 1) * sizeof(fftw_complex));

    cufftPlan2d(&forward, fftH, fftW, CUFFT_R2C);
    cufftPlan2d(&inverse, fftH, fftW, CUFFT_C2R);

    cudaMemcpy(dKernel, kernel, kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dSrc, src, height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dPadData, 0, fftH * fftW * sizeof(float));
    cudaMemset(dPadKern, 0, fftH * fftW * sizeof(float));

    padKernel(kernelW, kernelH, fftH, fftW, dKernel, dPadKern, kernelX, kernelY);
    padData(fftW, fftH, width, height, kernelX, kernelY, dSrc, dPadData);

    cufftExecR2C(forward, (cufftReal *)dPadKern, (cufftComplex *)dKS);
    cufftExecR2C(forward, (cufftReal *)dPadData, (cufftComplex *)dDS);



    // stbi_write_png("image_filter.png", width, height, val, dst, width*val);
	stbi_image_free(src);
	// stbi_image_free(dst);
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

