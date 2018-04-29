#include <stdio.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <sys/time.h>
// #include <fftw.h>
#include "fftw3.h"
#include "omp.h"

#include <stdlib.h>
#include <assert.h>
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

void padData(int fftW, int fftH, int width, int height, int kernelX, int kernelY, uint8_t* sData, double* data){
	int x = 0;
	int y = 0;
	int h = height+kernelY;
	int w = width +kernelX;

	int diffy = 0;
	int diffx = 0;
	int dx = 0;
	int dy = 0;
	// #if OMP
	// #pragma omp parallel for
	for (y = 0; y < fftH; y++){
		for (x = 0; x < fftW; x++){
			if (y < height){
				diffy = y;
			}
			if (x < width){
				diffx = x;
			}
			if (y>=height && y < fftH){
				diffy = h-1;
			}
			if (x>= width && x < fftW){
				diffx = w-1;
			}
			if (y>= fftH){
				diffy = 0;
			}
			if (x >=fftW){
				diffx = 0;
			}
			data[(y*fftW)+x] = (double)sData[(diffy*width)+diffx]; 
		}
	}
	//#endif
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
    fprintf(stderr, "ffth:%d fftw:%d\n", fftH, fftW);
  
    uint8_t* dst= (uint8_t*)malloc(width*height*sizeof(double));

    double* kernel = malloc(kernelW*kernelH*sizeof(double));
    memset(kernel, 0, sizeof(double)*kernelW*kernelH);
  // double kernelS[9]={(double)0, (double)-1, (double)0, (double)-1, (double)5, (double)-1, (double)0, (double)-1, (double)0};
    // memset(kernel,-1,sizeof(double));
    // memset(kernel+1,-1,sizeof(double));
    // memset(kernel+2,-1,sizeof(double));
    // memset(kernel+3,-1,sizeof(double));
    // memset(kernel+4,8,sizeof(double));
    // memset(kernel+5,-1,sizeof(double));
    // memset(kernel+6,-1,sizeof(double));
    // memset(kernel+7,-1,sizeof(double));
    // memset(kernel+8,-1,sizeof(double));
    int i;
    for(i=0;i<kernelW*kernelH;i++){
     	kernel[i]=0.1111111111111111f;
    }
   // kernel[4]=1.0f;

    //KERNEL
    double* fftKernel = malloc(fftW*fftH*sizeof(double));
    memset(fftKernel, 0, sizeof(double)*fftH*fftW);

    fftw_complex* fftKernelFT = fftw_malloc(fftW*fftH*sizeof(fftw_complex));
    memset(fftKernelFT, 0, sizeof(fftw_complex)*fftH*fftW);

    fftw_plan forwardK;	
    forwardK = fftw_plan_dft_r2c_2d(fftH, fftW, fftKernel, fftKernelFT, 0);
    
    padKernel(kernelW, kernelH, fftH, fftW, (double*)kernel, (double*)fftKernel, kernelX, kernelY);

    fftw_execute(forwardK);
    //END KERNEL

    double* dest = malloc(fftW*fftH*sizeof(double));
	memset(dest, 0, sizeof(double)*fftH*fftW);

	//DATASRC
   	double* fftSrc =malloc(fftW*fftH*sizeof(double));
	fftw_complex* fftDst =fftw_malloc(fftW*fftH*sizeof(fftw_complex));
	memset(fftSrc, 0, sizeof(double)*fftH*fftW);
	memset(fftDst, 0, sizeof(fftw_complex)*fftH*fftW);
	//MULT
	fftw_complex* multOut = fftw_malloc(fftW*fftH*sizeof(fftw_complex));
    memset(multOut, 0, sizeof(fftw_complex)*fftH*fftW);
	fftw_plan inverse;
    inverse=fftw_plan_dft_c2r_2d(fftH, fftW, multOut,dest,0);
    

    fftw_plan forwardD;	
    forwardD = fftw_plan_dft_r2c_2d(fftH, fftW, fftSrc, fftDst, 0);

    padData(fftW, fftH, width, height, kernelX, kernelY,  src, fftSrc);


    //iteration start
    int it = 0;
    for (it = 0; it < iter; it++){
	    fftw_execute(forwardD);

	    mult(fftKernelFT, fftDst, multOut, fftW, fftH);
	    fftw_execute(inverse);
	    memcpy(fftSrc, dest, sizeof(double)*fftW*fftH);
	}
    //end iteration
    
    int j = 0;
    for(i = 0; i < width;i++){
    	for(j = 0; j < height;j++){
    		dst[i*width+j] = (uint8_t)dest[i*fftW+j];
    	}
    }
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