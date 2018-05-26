// INCLUDES
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <time.h>

// FUNCTION DEFINITIONS
__global__ void nn_diff(double* input,double* weight, double* output, int column_size);

// DEFINES
#define SIZE 8
#define COLUMN_SIZE 4
#define ROW_SIZE  2


#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stdout, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stdout, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


// OTHER FUNCTIONS
__global__ void nn_diff(double* input,double** weight, double** output){
	   int i = blockDim.x * blockIdx.x + threadIdx.x;
       int j = blockDim.y * blockIdx.y + threadIdx.y;
	   if (i < ROW_SIZE && j < COLUMN_SIZE)
       {
		   output[i][j] = (input[i] - weight[i][j])*(input[i] - weight[i][j]);
	   }
       		   printf("i:%d j:%d out:%f in:%f wt:%f\n",i, j, output[i][j], input[i], weight[i][j]);
}

/*__global__ void nn_diff_add(double* output, double* output_add, int column_size, int size){
	   int i = blockDim.x * blockIdx.x + threadIdx.x;
	   if(i <size){
		   for(int p = 0; p < column_size; p++){
			   output_add[i] += output[p+i*column_size];
			   printf("%d %d:%f %f\n",i, p, output[p+i*column_size], output_add[i]);
		   }
		}
}*/

__global__ void nn_diff_add(double* output, double* output_add, int column_size, int size){
	/*extern*/ __shared__ double sdata[SIZE];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	sdata[tid] = output[i];
	__syncthreads();

	for (unsigned int s=1; s<blockDim.x; s*=2){
		if(tid% (2*s) == 0){
			sdata[tid] += sdata[tid+s];
		}
		__syncthreads();
	}

	if (tid==0) output_add[blockIdx.x]=sdata[0];
}
__device__ int nn_find_minimum(double* output_add){
	double min = 9999999;
	int min_loc = -1;
	for (int idx=0; idx < SIZE/COLUMN_SIZE; idx++){
		if(output_add[idx] < min){
			min = output_add[idx];
			min_loc = idx;
		}
	}
	return min_loc;
}
__global__ void nn_weight_update(double* input, double* weight, double learning_rate, double* output_add, int column_size, int size){
		int location = nn_find_minimum(output_add);
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if(i < size){
		   int idx = location*column_size+i;
		   printf("i:%d location:%d column_size:%d weight:%f index:%d input:%f\n", i, location, column_size, idx, weight[idx], input[i]);
		   weight[idx] = weight[idx] + learning_rate * (input[i] - weight[location*column_size+i]);
		   printf("i:%d location:%d column_size:%d weight:%f index:%d input:%f\n", i, location, column_size, idx, weight[idx], input[i]);
	   }
}



// MAIN FUNCTION
int main(){/*
	// VARIABLES
	double* input;
	double* weight;
	double* output;

	// Allocate Variables
	int in_size = 4;
	input  = (double*) malloc(COLUMN_SIZE*sizeof(double));
	weight = (double*) malloc(SIZE*sizeof(double));
	output = (double*) malloc(in_size*sizeof(double));

	for (	int idx=1; idx < COLUMN_SIZE; idx++){
		 input[idx] = rand() % 10;
	}
	for (	int idx=1; idx < SIZE; idx++){
			weight[idx] = rand() % 10;
	}
	for (	int idx=1; idx < in_size; idx++){
			output[idx] = rand() % 10;
	}*/
	double input[4] = {1.0, 1.0, 0.0, 0.0};
	double weight[2][4] = {{0.2,0.6,0.5,0.9},{0.8,0.4,0.7,0.3}};
	double output[2][4] = {0.0};
	double output_add[2] = {0.0};
	double learning_rate = 0.6;
	// Reset the GPUs
	cudaDeviceReset();

	// GPU Variable Declaration
	double *dev_input,**dev_weight,**dev_output, **dev_output_add;

	// GPU Variable Allocation
	cudaMalloc(&dev_input , sizeof(input));
	cudaMalloc(&dev_weight, sizeof(weight));
	cudaMalloc(&dev_output, sizeof(output));
	cudaMalloc(&dev_output_add, sizeof(output_add));
	cudaCheckErrors("cudamalloc fail");

	// Copy CPU Variable to GPU
	cudaMemcpy(dev_input,  input , sizeof(input), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_weight, weight, sizeof(weight), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_output, output, sizeof(output), cudaMemcpyHostToDevice);
	cudaCheckErrors("cuda memcpy fail");

    dim3 thrd_per_block(2,4);
    dim3 num_of_block(ROW_SIZE/thrd_per_block.x, SIZE/thrd_per_block.y);
	nn_diff<<< num_of_block ,thrd_per_block >>>(dev_input,dev_weight,dev_output); // output = (input - weight)^2
//	nn_diff_add<<< ROW_SIZE-1,1 >>>(dev_output,dev_output_add,COLUMN_SIZE,ROW_SIZE); //output_add = Addition of all the columns in a row
//	nn_weight_update<<< COLUMN_SIZE ,1 >>>(dev_input,dev_weight,learning_rate, dev_output_add, COLUMN_SIZE,COLUMN_SIZE); //output_add = Addition of all the columns in a row
	cudaMemcpy(output, dev_output ,sizeof(output), cudaMemcpyDeviceToHost);
//	cudaCheckErrors("cudamemcpy or cuda kernel fail");

	for(int idx=0; idx < SIZE; idx++){
		if (idx % COLUMN_SIZE == 0)
			printf("\n");
		printf("%f ",output[idx%COLUMN_SIZE][idx/COLUMN_SIZE]);
	}
	printf("\n");
	cudaFree(dev_input);
	cudaFree(dev_weight);
	cudaFree(dev_output);
	cudaFree(dev_output_add);
	return 0;
}
