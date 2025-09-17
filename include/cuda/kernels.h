#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

// evolution kernel
__global__ void evolveKernel(
    bool* current_grid;
    bool* next_grid;
    int grid_width;
    int grid_height;
    int grid_depth;
    int birth_min;
    int birth_max;
    int survival_min;
    int survival_max;
);

// util func to copy kernels


#endif  // KERNELS_H