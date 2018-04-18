#include <stdio.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <sys/time.h>
#include "mpi.h"

int main(int argc, char** argv) {
	int process_id, num_processes;
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
	
	/* Count time */ 
	uint64_t c = micro_time(); 
	int fd, width, height, val, iter;
	uint8_t* src;
	int bpp;
	if (process_id==0){
		setParams(argc, argv, &width, &height, &val, &iter);	

	    src = stbi_load(argv[1], &width, &height, &bpp, val);
		fprintf(stderr, "H:%d w:%d iterL%d \n", height, width, iter);
	}
	
	uint8_t* dst= malloc(width*height*val);
	
	MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&val, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&src, (width*height*val), MPI_UINT8_T, 0, MPI_COMM_WORLD);
	
    //need to make sure that has to be of size 2 or greater
	int start = start_position(height, num_processes, process_id);
	int start_pos=start*width*val;
	int end = height;
	if (process_id != num_processes-1){
		end = start_position(height, num_processes, process_id+1);
	}
	int end_pos=end*width*val;

	float kernel[3][3] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
	int i = 0;
	if (val == 1){
		int r = 0;
		int c = 0;
		int kr = 0;
		int kc = 0;

	 	int x = 0;
		int y = 0;
		int i = 0;
		for (i = 0; i < iter; i++){
			for (x = start; x < end; x++){
				for (y = 0; y < width; y++){
					float total = 0;
					for (r = x-1, kr = 0 ; r <= x+1 ; r++, kr++){
						for (c = y-3, kc = 0 ; c <= y+3 ; c+=3, kc++){
							if (width*r+c >= 0){
								int pixel = src[width*r+c];
								float sum = 0;
								sum =  ((float)pixel* kernel[kr][kc]);
								total += sum;
							}
						}
					}
					src[width * x + y] = total;
				}
			}
			MPI_Request req1;
			MPI_Request req2;
			if (process_id != 0){ //sending up
				MPI_Isend(src + start_pos, width, MPI_UINT8_T, process_id-1, 1, MPI_COMM_WORLD, &req1);
			}
			if (process_id != num_processes-1){ //send down
				MPI_Isend(src + end_pos-width, width, MPI_UINT8_T, process_id+1, 2, MPI_COMM_WORLD, &req2);
			}
			if (process_id != 0){ //getting from  up
				MPI_Irecv(src + start_pos-width, width, MPI_UINT8_T, process_id-1, 2, MPI_COMM_WORLD, &req2);
				MPI_Wait(&req2, MPI_STATUS_IGNORE);
			}
			if (process_id != num_processes-1){ //getting from down down
				MPI_Irecv(src + end_pos, width, MPI_UINT8_T, process_id+1, 1, MPI_COMM_WORLD, &req1);
				MPI_Wait(&req2, MPI_STATUS_IGNORE);
			}
		}
	}

	//combine and send to process 0
	memcpy(dst+start_pos, src+start_pos, sizeof(uint8_t)*width*val*(start-end));

	//do if process 0
	if (process_id == 0){
		stbi_write_png("image_filter.png", width, height, val, dst, width*val);
		stbi_image_free(src);
		stbi_image_free(dst);
	}
	/* compute time */
	c = micro_time() - c;
	double million = 1000 * 1000;
	fprintf(stdout, "Execution time: %.3f sec\n", c / million);

	return 0;
}


