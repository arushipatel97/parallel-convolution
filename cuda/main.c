#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "cudaConvolve.h"
#include "cudaFunc.h"
#include <sys/time.h>

int main(int argc, char** argv) {
	/* Count time */ 
	uint64_t c = micro_time(); 
	int fd, width, height, iter;
	
	int val = 1;
	int bpp;
	setParams(argc, argv, &width, &height, &val, &iter);	

    uint8_t* src = stbi_load(argv[1], &width, &height, &bpp, val);
	fprintf(stderr, "H:%d W:%d\n", height, width);

	size_t bytes = (val == 1) ? height * width : height * width*3;	
	
	uint8_t* dst= malloc(width*height*val);
	gpuConvolve(src, width, height, val, bytes, iter);

	stbi_write_png("image_filter.png", width, height, val, src, width*val);
	stbi_image_free(src);
	stbi_image_free(dst);
	/* compute time */
	c = micro_time() - c;
	double million = 1000 * 1000;
	fprintf(stdout, "Execution time: %.3f sec\n", c / million);
	return 0;
}
