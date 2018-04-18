#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cudaFunc.h"
#include <assert.h>

void setParams(int argc, char **argv, int *width, int *height, int *val, int *iter) {
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
		return;
	}
}

uint64_t micro_time(void) {
	struct timeval tv;
	assert(gettimeofday(&tv, NULL) == 0);
	return tv.tv_sec * 1000 * 1000 + tv.tv_usec;
}