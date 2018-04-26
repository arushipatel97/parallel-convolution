#include <stdio.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <sys/time.h>
#include "mpi.h"
#include "omp.h"

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
	}
	
	uint8_t* dst= (uint8_t*)malloc(width*height*val*sizeof(uint8_t));
	
	MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&val, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
    src = stbi_load(argv[1], &width, &height, &bpp, val);
	int start = start_position(height, num_processes, process_id);
	int start_pos=start*width*val;
	int end = height;
	if (process_id != num_processes-1){
		end = start_position(height, num_processes, process_id+1);
	}
	int end_pos=end*width*val;
	uint8_t* temp = (uint8_t*)malloc(width*height*val*sizeof(uint8_t));
	uint8_t* final = (uint8_t*)malloc(width*height*val*sizeof(uint8_t));
	uint8_t* finalDsp = (uint8_t*)malloc(width*height*val*sizeof(uint8_t));

	// fprintf(stderr, "id:%d start_pos:%d end_pos:%d\n", process_id, start_pos, end_pos);

	//float kernel[3][3] = {{1/9, 1/9, 1/9}, {1/9, 1/9, 1/9}, {1/9, 1/9, 1/9}};
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
		#pragma omp parallel for collapse(3)
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
					dst[width * x + y] = total;
				}
			}
			MPI_Request req1;
			MPI_Request req2;
			if (process_id != 0){ //sending up
				MPI_Isend(dst + start_pos, width, MPI_UINT8_T, process_id-1, 1, MPI_COMM_WORLD, &req1);
			}
			if (process_id != num_processes-1){ //send down
				MPI_Isend(dst + end_pos-width, width, MPI_UINT8_T, process_id+1, 2, MPI_COMM_WORLD, &req2);
			}
			if (process_id != 0){ //getting from  up
				MPI_Irecv(dst + start_pos-width, width, MPI_UINT8_T, process_id-1, 2, MPI_COMM_WORLD, &req2);
				MPI_Wait(&req2, MPI_STATUS_IGNORE);
			}
			if (process_id != num_processes-1){ //getting from down down
				MPI_Irecv(dst + end_pos, width, MPI_UINT8_T, process_id+1, 1, MPI_COMM_WORLD, &req1);
				MPI_Wait(&req2, MPI_STATUS_IGNORE);
			}
			src = dst;
		}

		memset(final, 0, sizeof(uint8_t)*width*height);
		memset(finalDsp, 0, sizeof(uint8_t)*width*height);
		memcpy(final+(start*width), src+(start*width), sizeof(uint8_t)*(end-start)*width);
		MPI_Reduce(final, finalDsp, width*height, MPI_UINT8_T, MPI_SUM, 0, MPI_COMM_WORLD);
	}else{
		int rr = 0;
		int c = 0;
		int kr = 0;
		int kc = 0;

	 	int x = 0;
		int y = 0;
		int i = 0;
		#pragma omp parallel for collapse(3)
		for (i = 0; i < iter; i++){
			for (x = start; x < end; x++){
				for (y = 0; y < width; y++){
					float r = 0;
					float g = 0;
					float b = 0;
					for (rr = x-1, kr = 0 ; rr <= x+1 ; rr++, kr++){
						for (c = (y*3)-3, kc = 0 ; c <= (y*3)+3 ; c+=3, kc++){
							if (width*3*rr+c >= 0){
								int pixelR = src[width*3 * rr + c];
								int pixelG = src[width *3* rr + c+1];
								int pixelB = src[width *3* rr + c+2];
								r += ((float)pixelR* kernel[kr][kc]);
								g += ((float)pixelG* kernel[kr][kc]);
							    b += ((float)pixelB * kernel[kr][kc]);
							}
						}
					}
					dst[width *3* x + (3*y)] = r;
					dst[width *3* x + (3*y)+1] = g;
					dst[width *3* x + (3*y)+2] = b;
				}
			}
			MPI_Request req1;
			MPI_Request req2;
			if (process_id != 0){ //sending up
				MPI_Isend(dst + start_pos, width*3, MPI_UINT8_T, process_id-1, 1, MPI_COMM_WORLD, &req1);
			}
			if (process_id != num_processes-1){ //send down
				MPI_Isend(dst + end_pos-(3*width), width*3, MPI_UINT8_T, process_id+1, 2, MPI_COMM_WORLD, &req2);
			}
			if (process_id != 0){ //getting from  up
				MPI_Irecv(dst + start_pos-(width*3), (3*width), MPI_UINT8_T, process_id-1, 2, MPI_COMM_WORLD, &req2);
				MPI_Wait(&req2, MPI_STATUS_IGNORE);
			}
			if (process_id != num_processes-1){ //getting from down down
				MPI_Irecv(dst + end_pos, 3*width, MPI_UINT8_T, process_id+1, 1, MPI_COMM_WORLD, &req1);
				MPI_Wait(&req2, MPI_STATUS_IGNORE);
			}
			src = dst;
		}
		memset(final, 0, sizeof(uint8_t)*width*height*val);
		memset(finalDsp, 0, sizeof(uint8_t)*width*height*val);
		memcpy(final+(start*width*val), src+(start*width*val), sizeof(uint8_t)*(end-start)*width*val);
		MPI_Reduce(final, finalDsp, width*height*val, MPI_UINT8_T, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	if (process_id == 0){
		stbi_write_png("image_filter.png", width, height, val, finalDsp, width*val);	
		/* compute time */
		c = micro_time() - c;
		double million = 1000 * 1000;
		fprintf(stdout, "Execution time: %.3f sec\n", c / million);
	}
	MPI_Finalize();
	return 0;
}


