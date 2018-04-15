#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cudaFunc.h"

void setParams(int argc, char **argv, int *width, int *height, int *val) {
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
		return;
	}
}

