#include <stdio.h>
#include <vector>
#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <thrust/count.h>

#define THREADS_PER_BLOCK 256

// assign random numbers to each node
int* init_rank(int node_cnt) {
    int* rank = (int*) malloc(sizeof(int) * node_cnt);
    for (int i = 0; i < node_cnt; ++i) {
        rank[i] = rand();
    }
    return rank;
}

__global__ void jones_plassmann_kernel() {
    // TODO
}

void jones_plassmann(int node_cnt, int edge_cnt, int* colors, int *nbrs_start, int *nbrs) {
    // initialization
    int num_blocks = (node_cnt + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int* rank = init_rank(node_cnt);

    // allcoate memory
    int* device_nbrs;
    int* device_nbrs_start;
    int* device_colors;
    cudaMalloc(&device_nbrs, edge_cnt * 2);
    cudaMalloc(&device_nbrs_start, node_cnt + 1);
    cudaMalloc(&device_colors, node_cnt);
    // copy input to device
    cudaMemcpy(device_colors, colors, node_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(device_nbrs, nbrs, edge_cnt * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(device_nbrs_start, nbrs_start, node_cnt + 1, cudaMemcpyHostToDevice);

    for (int i = 0; i < node_cnt; ++i) {
        jones_plassmann_kernel<<<num_blocks, THREADS_PER_BLOCK>>>();
        int uncolored = (int)thrust::count(colors, colors + node_cnt, 0);
        if (uncolored == 0) break;
    }

    // copy result back
    cudaMemcpy(colors, device_colors, node_cnt, cudaMemcpyDeviceToHost);
    // free memory
    cudaFree(device_nbrs);
    cudaFree(device_nbrs_start);
    cudaFree(device_colors);
}

void printCudaInfo() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}