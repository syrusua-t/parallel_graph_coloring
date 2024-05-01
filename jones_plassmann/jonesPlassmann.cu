#include <stdio.h>
#include <vector>
#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <thrust/count.h>

#include "mode.h"
#include "predictor.cpp"

void check_error(std::string s) {
    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "%s: WARNING: A CUDA error occured: code=%d, %s\n", s.c_str(), errCode,
                cudaGetErrorString(errCode));
        exit(1);
    }
}

// assign random numbers to each node
void init_rank(int* rank, int node_cnt) {
    for (int i = 0; i < node_cnt; ++i) {
        rank[i] = rand();
    }
}


// instead of precomputing random numbers, use a hash function to compute priority on the fly
// source: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key (could find better candidates)
__device__ inline int hash(int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

__global__ void jones_plassmann_basic_kernel(int cur_color, int node_cnt, 
    int* colors, int *nbrs_start, int *nbrs, int* rank) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    // already colored, skip
    if (node >= node_cnt || colors[node] != 0) return;
    
    bool is_max = true;
    for (int nbr_idx = nbrs_start[node]; nbr_idx < nbrs_start[node + 1]; ++nbr_idx) {
        int nbr = nbrs[nbr_idx];
        // ignore colored neighbor
        if (colors[nbr] != 0 && colors[nbr] != cur_color) {
            continue;
        }
        if (rank[node] <= rank[nbr]) is_max = false;
    }
    if (is_max) colors[node] = cur_color;
}

__global__ void jones_plassmann_basic_kernel_hash(int cur_color, int node_cnt, 
    int* colors, int *nbrs_start, int *nbrs) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    // already colored, skip
    if (node >= node_cnt || colors[node] != 0) return;
    int my_hash = hash(node);
    bool is_max = true;
    for (int nbr_idx = nbrs_start[node]; nbr_idx < nbrs_start[node + 1]; ++nbr_idx) {
        int nbr = nbrs[nbr_idx];
        // ignore colored neighbor
        if (colors[nbr] != 0 && colors[nbr] != cur_color) {
            continue;
        }
        if (my_hash <= hash(nbr)) is_max = false;
    }
    if (is_max) colors[node] = cur_color;
}

__global__ void jones_plassmann_minmax_kernel(int cur_color, int node_cnt, 
    int* colors, int *nbrs_start, int *nbrs, int* rank) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    // already colored, skip
    if (node >= node_cnt || colors[node] != 0) return;
    
    bool is_max = true;
    bool is_min = true;
    for (int nbr_idx = nbrs_start[node]; nbr_idx < nbrs_start[node + 1]; ++nbr_idx) {
        int nbr = nbrs[nbr_idx];
        // ignore colored neighbor
        if (colors[nbr] != 0 && colors[nbr] != cur_color && colors[nbr] != cur_color + 1) {
            continue;
        }
        if (rank[node] <= rank[nbr]) is_max = false;

        if (rank[node] >= rank[nbr]) is_min = false;
    }
    if (is_max) colors[node] = cur_color;
    if (is_min) colors[node] = cur_color + 1;
}

__global__ void jones_plassmann_multihash_kernel(int cur_color, int node_cnt, 
    int* colors, int *nbrs_start, int *nbrs, int hash_cnt) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    // already colored, skip
    if (node >= node_cnt || colors[node] != 0) return;
    long is_max = -1;
    for (int nbr_idx = nbrs_start[node]; nbr_idx < nbrs_start[node + 1]; ++nbr_idx) {
        int nbr = nbrs[nbr_idx];
        // ignore colored neighbor
        if (colors[nbr] != 0 && colors[nbr] < cur_color) {
            continue;
        }
        int nbr_hash = hash(nbr);
        int my_hash = hash(node);
        for (int i = 0; i < hash_cnt; ++i) {
            if (my_hash <= nbr_hash) is_max = is_max & (~(1 << i));
            nbr_hash = hash(nbr_hash);
            my_hash = hash(my_hash);
        }
    }
    long mask = 1;
    for (int i = 0; i < hash_cnt; ++i) {
        if (is_max & (mask << i)) {
            colors[node] = cur_color + i;
            break;
        }
    }
}


void jones_plassmann(int node_cnt, int edge_cnt, int* colors, int *nbrs_start, int *nbrs, Mode mode) {
    // initialization
    int num_blocks = (node_cnt + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int* rank = (int*)malloc(sizeof(int) * node_cnt); 
    init_rank(rank, node_cnt);

    // allcoate memory
    int* device_nbrs;
    int* device_nbrs_start;
    int* device_colors;
    int* device_rank;
    cudaMalloc(&device_nbrs, sizeof(int) * (edge_cnt * 2));
    cudaMalloc(&device_nbrs_start, sizeof(int) * (node_cnt + 1));
    cudaMalloc(&device_colors, sizeof(int) * node_cnt);
    cudaMalloc(&device_rank, sizeof(int) * node_cnt);
    // copy input to device
    cudaMemcpy(device_rank, rank, sizeof(int) * node_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(device_colors, colors, sizeof(int) * node_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(device_nbrs, nbrs, sizeof(int) * (edge_cnt * 2), cudaMemcpyHostToDevice);
    cudaMemcpy(device_nbrs_start, nbrs_start, sizeof(int) * (node_cnt + 1), cudaMemcpyHostToDevice);

    
    switch (mode) {
        case Basic:
            for (int cur_color = 1; cur_color <= node_cnt; ++cur_color) {
                    jones_plassmann_basic_kernel<<<num_blocks, THREADS_PER_BLOCK>>>
                    (cur_color, node_cnt, device_colors, device_nbrs_start, device_nbrs, device_rank);
                    cudaMemcpy(colors, device_colors, sizeof(int) * node_cnt, cudaMemcpyDeviceToHost);
                int uncolored = (int)thrust::count(colors, colors + node_cnt, 0);
                if (uncolored == 0) break;
            }
            break;
        case MinMax:
            for (int cur_color = 1; cur_color <= 2 * node_cnt; cur_color += 2) {
                    jones_plassmann_minmax_kernel<<<num_blocks, THREADS_PER_BLOCK>>>
                    (cur_color, node_cnt, device_colors, device_nbrs_start, device_nbrs, device_rank);
                    cudaMemcpy(colors, device_colors, sizeof(int) * node_cnt, cudaMemcpyDeviceToHost);
                int uncolored = (int)thrust::count(colors, colors + node_cnt, 0);
                if (uncolored == 0) break;
            }
            break;
        case MultiHash:
            {
                int prev_uncolored = node_cnt;
                Predictor p(CONSTANT, edge_cnt, node_cnt);
                int hash_cnt = p.get_hash_cnt();
                for (int cur_color = 1; cur_color <= hash_cnt * node_cnt; cur_color += hash_cnt) {
                    jones_plassmann_multihash_kernel<<<num_blocks, THREADS_PER_BLOCK>>>
                        (cur_color, node_cnt, device_colors, device_nbrs_start, device_nbrs, hash_cnt);
                        cudaMemcpy(colors, device_colors, sizeof(int) * node_cnt, cudaMemcpyDeviceToHost);
                    int uncolored = (int)thrust::count(colors, colors + node_cnt, 0);
                    if (uncolored == 0) break;
                    p.update_hash_cnt(((float)prev_uncolored - uncolored)/(float)prev_uncolored);
                    hash_cnt = p.get_hash_cnt();
                }
                break;
            }
        case BasicOpt:
            for (int cur_color = 1; cur_color <= node_cnt; ++cur_color) {
                    jones_plassmann_basic_kernel_hash<<<num_blocks, THREADS_PER_BLOCK>>>
                    (cur_color, node_cnt, device_colors, device_nbrs_start, device_nbrs);
                    cudaMemcpy(colors, device_colors, sizeof(int) * node_cnt, cudaMemcpyDeviceToHost);
                int uncolored = (int)thrust::count(colors, colors + node_cnt, 0);
                if (uncolored == 0) break;
            }
            break;
        default:
            printf("\033[1;31m ERROR: Unrecognized coloring mode!\033[0m\n");
            exit(1);
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