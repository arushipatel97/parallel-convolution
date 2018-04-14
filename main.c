#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "serial.h"

int main(int argc, char** argv) {
	int fd, width, height, loops;
	char *image;
	color_t imageType;

	int bpp;
	setParams(argc, argv, &image, &width, &height, &imageType);	
	int val = 1;
	if (imageType != GREY){
		val = 3;
	}

    uint8_t* src = stbi_load("image.png", &width, &height, &bpp, val);
	
	fprintf(stderr, "H:%d w:%d\n", height, width);

	
	size_t bytes = (imageType == GREY) ? height * width : height * width*3;	

	if (src == NULL){
		fprintf(stderr, "%s\n", "UGGH");
	}
	
	uint8_t* dst= malloc(width*height*val);
	//memcpy(dst, src, bytes*sizeof(uint8_t));
	
	if (imageType == GREY){
		convolve_G(src, dst, width, height, imageType);
	}
	else{
	    convolve_RGB(src, dst, width, height, imageType);
	}
	stbi_write_png("imageblur.png", width, height, val, dst, width*val);
	return EXIT_SUCCESS;
}