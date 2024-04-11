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

void check_error(std::string s) {
    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "%s: WARNING: A CUDA error occured: code=%d, %s\n", s.c_str(), errCode,
                cudaGetErrorString(errCode));
        exit(1);
    }
}

// assign random numbers to each node
int* init_rank(int node_cnt) {
    int* rank = (int*) malloc(sizeof(int) * node_cnt);
    for (int i = 0; i < node_cnt; ++i) {
        rank[i] = rand();
    }
    return rank;
}

__global__ void jones_plassmann_kernel(int cur_color, int node_cnt, 
    int* colors, int *nbrs_start, int *nbrs, int* rank) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    // already colored, skip
    if (node >= node_cnt || colors[node] != 0) return;
    
    bool is_max = true;
    for (int nbr_idx = nbrs_start[node]; nbr_idx < nbrs_start[node + 1]; ++nbr_idx) {
        int nbr = nbrs[nbr_idx];
        // ignore colored neighbor
        if (colors[nbr] != 0) {
            continue;
        }
        if (rank[node] <= rank[nbr]) is_max = false;
    }
    if (is_max) colors[node] = cur_color;
}

void jones_plassmann(int node_cnt, int edge_cnt, int* colors, int *nbrs_start, int *nbrs) {
    // initialization
    int num_blocks = (node_cnt + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int* rank = init_rank(node_cnt);

    // allcoate memory
    int* device_nbrs;
    int* device_nbrs_start;
    int* device_colors;
    int* device_rank;
    cudaMalloc(&device_nbrs, edge_cnt * 2);
    cudaMalloc(&device_nbrs_start, node_cnt + 1);
    cudaMalloc(&device_colors, node_cnt);
    cudaMalloc(&device_rank, node_cnt);
    // copy input to device
    cudaMemcpy(device_rank, rank, node_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(device_colors, colors, node_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(device_nbrs, nbrs, edge_cnt * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(device_nbrs_start, nbrs_start, node_cnt + 1, cudaMemcpyHostToDevice);

    for (int cur_color = 1; cur_color <= node_cnt; ++cur_color) {
        jones_plassmann_kernel<<<num_blocks, THREADS_PER_BLOCK>>>
            (cur_color, node_cnt, device_colors, device_nbrs_start, device_nbrs, device_rank);
        cudaMemcpy(colors, device_colors, node_cnt, cudaMemcpyDeviceToHost);
        int uncolored = (int)thrust::count(colors, colors + node_cnt, 0);
        if (uncolored == 0) break;
    }
    
    // free memory
    cudaFree(device_rank);
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