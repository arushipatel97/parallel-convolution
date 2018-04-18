#include <stdio.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "serial.h"
#include <sys/time.h>

int main(int argc, char** argv) {
	/* Count time */ 
	uint64_t c = micro_time(); 
	int fd, width, height, val, iter;

	int bpp;
	setParams(argc, argv, &width, &height, &val, &iter);	

    uint8_t* src = stbi_load(argv[1], &width, &height, &bpp, val);
	fprintf(stderr, "H:%d w:%d iterL%d \n", height, width, iter);
	
	uint8_t* dst= malloc(width*height*val);
	
	if (val == 1){
		convolve_G(src, dst, width, height, iter);
	}
	else{
	    convolve_RGB(src, dst, width, height, iter);
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