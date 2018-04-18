#include <stdio.h>
#include <string.h>
#include "serial.h"
#include <stdlib.h>
#include <assert.h>
#include "omp.h"

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


void conv_RGB(uint8_t*src, uint8_t* dst, int x, int y, int width, int height, float* save){
	float div = 9;
    //float kernel[3][3] = {{0/div, 0/div, 0/div}, {0/div, 1/div, 0/div}, {0/div, 0/div, 0/div}};
    //float kernel[3][3] = {{1/div, 1/div, 1/div}, {1/div, 1/div, 1/div}, {1/div, 1/div, 1/div}};
    float kernel[3][3] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
	int rr = 0;
	int c = 0;
	int kr = 0;
	int kc = 0;
	float r = 0, g = 0, b = 0;
	#pragma omp parallel for schedule
	for (rr = x-1, kr = 0 ; rr <= x+1 ; rr++, kr++){
		for (c = y-3, kc = 0 ; c <= y+3 ; c+=3, kc++){
			if (width*rr+c >= 0){
				int pixelR = src[width * rr + c];
				int pixelG = src[width * rr + c+1];
				int pixelB = src[width * rr + c+2];
				float sumR =0;
				float sumG = 0;
				float sumB = 0;
				// if (save[pixelR]){
				// 	sumR = save[pixelR];
				// }
				// else{
					sumR = ((float)pixelR* kernel[kr][kc]);
				// }
				r += sumR;
				// if (save[pixelG]){
				// 	sumG = save[pixelG];
				// }
				// else{
					sumG = ((float)pixelG* kernel[kr][kc]);
				// }
				g += sumG;
				// if (save[pixelB]){
				// 	sumB = save[pixelB];
				// }
				// else{
					sumB = ((float)pixelB * kernel[kr][kc]);
				// }
				b += sumB;
			}
		}
	}
	dst[width * x + y] = r;
	dst[width * x + y+1] = g;
	dst[width * x + y+2] = b;
}

	
void convolve_RGB(uint8_t* src, uint8_t* dst, int width, int height, int iter){
	float save[256];
	memset(save, 0, sizeof(float)*256);
	int x = 0;
	int y = 0;
	int i = 0;
	for (i = 0; i < iter; i++){
		// #pragma omp parallel for schedule 
		for (x = 0; x < height; x++){
			for (y = 0; y < width; y++){
				conv_RGB(src, dst, x, y*3, width*3, height, save);
			}
		}
	}
}


void convolve_G(uint8_t* src, uint8_t*dst, int width, int height, int iter){
	float div = 9;
    //float kernel[3][3] = {{0/div, 0/div, 0/div}, {0/div, 1/div, 0/div}, {0/div, 0/div, 0/div}};
    float kernel[3][3] = {{1/div, 1/div, 1/div}, {1/div, 1/div, 1/div}, {1/div, 1/div, 1/div}};
    //float kernel[3][3] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
    float save[256];
    memset(save, 0, sizeof(float)*256);
	int r = 0;
	int c = 0;
	int kr = 0;
	int kc = 0;

 	int x = 0;
	int y = 0;
	int i = 0;
	for (i =0; i < iter; i++){
		#pragma omp parallel for schedule collapse(4)
		for (x = 0; x < height; x++){
			for (y = 0; y < width; y++){
				float total = 0;
				for (r = x-1, kr = 0 ; r <= x+1 ; r++, kr++){
					for (c = y-3, kc = 0 ; c <= y+3 ; c+=3, kc++){
						if (width*r+c >= 0){
							int pixel = src[width*r+c];
							float sum = 0;
							// if (!save[pixel]){
								sum =  ((float)pixel* kernel[kr][kc]);
							// 	save[pixel] = sum;
							// }
							// else{
							// 	sum = save[pixel];
							// }
							total += sum;
						}
					}
				}
				dst[width * x + y] = total;
			}
		}
	}
}


uint64_t micro_time(void) {
	struct timeval tv;
	assert(gettimeofday(&tv, NULL) == 0);
	return tv.tv_sec * 1000 * 1000 + tv.tv_usec;
}