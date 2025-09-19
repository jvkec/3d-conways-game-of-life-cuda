#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

// evolution kernel
__global__ void evolveKernel(
    bool* current_grid,
    bool* next_grid,
    int grid_width,
    int grid_height,
    int grid_depth,
    int birth_min,
    int birth_max,
    int survival_min,
    int survival_max
);

// init kernel
__global__ void initialKernel(
    bool* grid,
    int width,
    int height,
    int depth,
    float density
);

// util func to copy kernels
__global__ void copyKernel(
    bool* src,
    bool* dst,
    int width, 
    int height, 
    int depth
);

#endif  // KERNELS_H