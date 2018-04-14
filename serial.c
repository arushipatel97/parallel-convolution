#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>
#include "serial.h"

void setParams(int argc, char **argv, char **image, int *width, int *height,  color_t *imageType) {
	if (argc == 5 && !strcmp(argv[4], "grey")) {
		*image = (char *)malloc((strlen(argv[1])+1) * sizeof(char));
		strcpy(*image, argv[1]);	
		*width = atoi(argv[2]);
		*height = atoi(argv[3]);
		*imageType = GREY;
	} else if (argc == 5 && !strcmp(argv[4], "rgb")) {
		*image = (char *)malloc((strlen(argv[1])+1) * sizeof(char));
		strcpy(*image, argv[1]);	
		*width = atoi(argv[2]);
		*height = atoi(argv[3]);
		*imageType = RGB;
	} else {
		fprintf(stderr, "Error Input!\n%s image_name width height [rgb/grey].\n", argv[0]);
		exit(EXIT_FAILURE);
	}
}

void convolve_RGB(uint8_t* src, uint8_t* dst, int width, int height){
	int div = 1;
    float f[3][3] = {{1/div, 1/div, 1/div}, {1/div, 1/div, 1/div}, {1/div, 1/div, 1/div}};
    int rows = height;
	int cols = width;
	int i = 0;
	int j = 0;
	int k = 0;
	int l = 0;
	float r = 0;
	float g = 0;
	float b = 0;
	float pixel = 0;
	for (i = 0; i < rows; i++){
		for (j = 0; j < cols; j++){
			r = 0;
			g = 0;
			b = 0;
			for(k = -1; k < 1; k++){
				for(l = -1; l < 1; l++){
					if (i > k && j >l){
						r += src[(rows*3*(i-k))+j] *f[k][l];
						g += src[(rows*3*(i-k))+j+1] *f[k][l];
						b += src[(rows*3*(i-k))+j+2] *f[k][l];
					}
				}
			}
		    dst[(i*cols)+(j*3)] = r;
		    dst[(i*cols)+(j*3)+1] = g;
		    dst[(i*cols)+(j*3)+2] = b;
		}
	}
}

void convolve_G(uint8_t* src, uint8_t*dst, int width, int height){
    float f[3][3] = {{1/9, 1/9, 1/9}, {2/9, 4/9, 2/9}, {1/9, 2/9, 1/9}};
    int rows = height;
	int cols = width;
	int i = 0;
	int j = 0;
	int k = 0;
	int l = 0;
	float total = 0;
	float pixel = 0;
	for (i = 0; i < rows; i++){
		for (j = 0; j < cols; j++){
			total = 0;
			for(k = -1; k < 1; k++){
				for(l = -1; l < 1; l++){
					if (i > k && j >l){
						pixel = src[(rows*(i-k))+(j-l)];
						total += f[k][l]*pixel;
					}
				}
			}
		    dst[(i*cols)+j] = total;
		}
	}
}

int write_all(int fd , uint8_t* buff , int size) {
	int n, sent;
	for (sent = 0 ; sent < size ; sent += n)
		if ((n = write(fd, buff + sent, size - sent)) == -1)
			return -1;
	return sent;
}

int read_all(int fd , uint8_t* buff , int size) {
	int n, sent;
	for (sent = 0 ; sent < size ; sent += n)
		if ((n = read(fd, buff + sent, size - sent)) == -1)
			return -1;
	return sent;
}
