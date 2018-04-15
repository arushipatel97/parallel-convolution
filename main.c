#include <stdio.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "serial.h"

int main(int argc, char** argv) {
	int fd, width, height, val;

	int bpp;
	setParams(argc, argv, &width, &height, &val);	

    uint8_t* src = stbi_load(argv[1], &width, &height, &bpp, val);
	fprintf(stderr, "H:%d w:%d\n", height, width);
	
	uint8_t* dst= malloc(width*height*val);
	
	if (val == 1){
		convolve_G(src, dst, width, height);
	}
	else{
	    convolve_RGB(src, dst, width, height);
	}
	stbi_write_png("image_filter.png", width, height, val, dst, width*val);
	stbi_image_free(src);
	stbi_image_free(dst);
	return 0;
}