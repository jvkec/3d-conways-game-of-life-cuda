#include "../../include/cuda/kernels.h"

// flatten3D is a helper function to flatten a 3D index into a 1D index
// forceinline because called often
__device__ __forceinline__ int flatten3D(int x, int y, int z, int width, int height) {
    return x + y * width + z * width * height;
}

// forceinline because called on every cell in the grid
__device__ __forceinline__ int countNeighbors(
    const bool* grid,
    int x, 
    int y, 
    int z,
    int width, 
    int height, 
    int depth
) 
{
    int neighbor_count = 0;
    for (int dz = -1; dz <= 1; ++dz) {
        int nz = z + dz;
        if (nz < 0 || nz >= depth) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            int ny = y + dy;
            if (ny < 0 || ny >= height) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx;
                if (nx < 0 || nx >= width) continue;
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nidx = flatten3D(nx, ny, nz, width, height);
                neighbor_count += grid[nidx] ? 1 : 0;
            }
        }
    }
    return neighbor_count;
}

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
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= grid_width || y >= grid_height || z >= grid_depth) {
        return;
    }

    int idx = flatten3D(x, y, z, grid_width, grid_height);
    int neighbors = countNeighbors(current_grid, x, y, z, grid_width, grid_height, grid_depth);
    bool alive = current_grid[idx];

    bool will_live;
    if (!alive) {
        will_live = (neighbors >= birth_min) && (neighbors <= birth_max);
    } else {
        will_live = (neighbors >= survival_min) && (neighbors <= survival_max);
    }

    next_grid[idx] = will_live;
}

__global__ void initialKernel(
    bool* grid,
    int width, 
    int height, 
    int depth,
    float density
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= width || y >= height || z >= depth) return;
    
    int thread_id = flatten3D(x, y, z, width, height);
    
    // lcg per-thread rng based on thread_id
    unsigned int seed = 1664525u * (thread_id + 1u) + 1013904223u;
    // convert to float in [0,1)
    float rnd = (seed & 0x00FFFFFF) / 16777216.0f;
    grid[thread_id] = (rnd < density);
}

__global__ void copyKernel(
    bool* src,
    bool* dst,
    int width, 
    int height, 
    int depth
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= width || y >= height || z >= depth) return;
    
    int thread_id = flatten3D(x, y, z, width, height);
    dst[thread_id] = src[thread_id];
}