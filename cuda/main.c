#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "cudaConvolve.h"
#include "cudaFunc.h"

int main(int argc, char** argv) {
	int fd, width, height;
	
	int val = 1;
	int bpp;
	setParams(argc, argv, &width, &height, &val);	

    uint8_t* src = stbi_load(argv[1], &width, &height, &bpp, val);
	fprintf(stderr, "H:%d W:%d\n", height, width);

	size_t bytes = (val == 1) ? height * width : height * width*3;	
	
	uint8_t* dst= malloc(width*height*val);
	gpuConvolve(src, width, height, val, bytes);

	stbi_write_png("image_filter.png", width, height, val, src, width*val);
	stbi_image_free(src);
	stbi_image_free(dst);
	return 0;
}
