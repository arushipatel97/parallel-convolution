#include <stdio.h>
#include <string.h>
#include "serial.h"

void setParams(int argc, char **argv, int *width, int *height, int* val) {
	if (argc == 5 && !strcmp(argv[4], "gray")) {
		*width = atoi(argv[2]);
		*height = atoi(argv[3]);
		*val = 1;
	} else if (argc == 5 && !strcmp(argv[4], "rgb")) {
		*width = atoi(argv[2]);
		*height = atoi(argv[3]);
		*val = 3;
	} else {
		fprintf(stderr, "Incorrect Input!\n");
	}
}


void conv_RGB(uint8_t*src, uint8_t* dst, int x, int y, int width, int height){
	float div = 9;
    //float kernel[3][3] = {{0/div, 0/div, 0/div}, {0/div, 1/div, 0/div}, {0/div, 0/div, 0/div}};
    //float kernel[3][3] = {{1/div, 1/div, 1/div}, {1/div, 1/div, 1/div}, {1/div, 1/div, 1/div}};
    float kernel[3][3] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
	int rr = 0;
	int c = 0;
	int kr = 0;
	int kc = 0;
	float r = 0, g = 0, b = 0;
	for (rr = x-1, kr = 0 ; rr <= x+1 ; rr++, kr++){
		for (c = y-3, kc = 0 ; c <= y+3 ; c+=3, kc++){
			if (width*rr+c >= 0){
				r += ((float)src[width * rr + c]* kernel[kr][kc]);
				g += ((float)src[width * rr + c+1] * kernel[kr][kc]);
				b += ((float)src[width * rr + c+2] * kernel[kr][kc]);
			}
		}
	}
	dst[width * x + y] = r;
	dst[width * x + y+1] = g;
	dst[width * x + y+2] = b;
}

	
void convolve_RGB(uint8_t* src, uint8_t* dst, int width, int height){
	int x = 0;
	int y = 0;
	for (x = 0; x < height; x++){
		for (y = 0; y < width; y++){
			conv_RGB(src, dst, x, y*3, width*3, height);
		}
	}
}


void convolve_G(uint8_t* src, uint8_t*dst, int width, int height){
	float div = 9;
    //float kernel[3][3] = {{0/div, 0/div, 0/div}, {0/div, 1/div, 0/div}, {0/div, 0/div, 0/div}};
    //float kernel[3][3] = {{1/div, 1/div, 1/div}, {1/div, 1/div, 1/div}, {1/div, 1/div, 1/div}};
    float kernel[3][3] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
	int r = 0;
	int c = 0;
	int kr = 0;
	int kc = 0;

 	int x = 0;
	int y = 0;
	for (x = 0; x < height; x++){
		for (y = 0; y < width; y++){
			float total = 0;
			for (r = x-1, kr = 0 ; r <= x+1 ; r++, kr++){
				for (c = y-3, kc = 0 ; c <= y+3 ; c+=3, kc++){
					if (width*r+c >= 0){
						total += ((float)src[width * r + c]* kernel[kr][kc]);
					}
				}
			}
			dst[width * x + y] = total;
		}
	}
}

