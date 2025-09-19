#include "cuda/game_logic.h"
#include "cuda/types.h"
#include "cuda/kernels.h"
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

// "default" constructor
CUDAGameOfLife::CUDAGameOfLife(
    int width, 
    int height, 
    int depth,                
    int birth_min, 
    int birth_max, 
    int survival_min, 
    int survival_max
)
{
    d_grid_current = nullptr;
    d_grid_next = nullptr;
    block_size = dim3(1, 1, 1);
    grid_size = dim3(1, 1, 1);
    
    // Create params with provided values (defaults used if not specified)
    params.width = width;
    params.height = height;
    params.depth = depth;
    params.birth_min = birth_min;
    params.birth_max = birth_max;
    params.survival_min = survival_min;
    params.survival_max = survival_max;
    
    calculateExecutionConfig();
    allocateDeviceMemory();
}

CUDAGameOfLife::CUDAGameOfLife(const GameOfLifeParams& game_params) 
{
    d_grid_current = nullptr;
    d_grid_next = nullptr;
    block_size = dim3(1, 1, 1);
    grid_size = dim3(1, 1, 1);
    params = game_params;
    
    calculateExecutionConfig();
    allocateDeviceMemory();
}


CUDAGameOfLife::~CUDAGameOfLife()
{
    freeDeviceMemory();
}

void CUDAGameOfLife::calculateExecutionConfig()
{
    // 512 threads per block
    block_size = dim3(8, 8, 8);

    // ceiling div incase grid not mult of block size
    grid_size = dim3(
        (params.width + block_size.x - 1) / block_size.x,
        (params.height + block_size.y - 1) / block_size.y,
        (params.depth + block_size.z - 1) / block_size.z
    );
    
    std::cout << "CUDA Config - Grid: " << grid_size.x << "x" << grid_size.y << "x" << grid_size.z << std::endl
              << "CUDA Config - Blocks: " << block_size.x << "x" << block_size.y << "x" << block_size.z << std::endl;
}

void CUDAGameOfLife::allocateDeviceMemory()
{
    size_t total_cells = getTotalCells();
    size_t memory_size = total_cells * sizeof(bool);

    cudaError_t err1 = cudaMalloc(&d_grid_current, memory_size);
    cudaError_t err2 = cudaMalloc(&d_grid_next, memory_size);
    
    if (err1 != cudaSuccess || err2 != cudaSuccess) 
    {
        std::cerr << "Failed to allocate device memory!" << std::endl;
        std::cerr << "Current grid error: " << cudaGetErrorString(err1) << std::endl;
        std::cerr << "Next grid error: " << cudaGetErrorString(err2) << std::endl;
        exit(1);
    }
    
    std::cout << "Allocated " << memory_size << " bytes per grid on GPU" << std::endl;
}

void CUDAGameOfLife::freeDeviceMemory()
{
    if (d_grid_current) 
    {
        cudaFree(d_grid_current);
        d_grid_current = nullptr;
    }
    if (d_grid_next)
    {
        cudaFree(d_grid_next);
        d_grid_next = nullptr;
    } 
    std::cout << "Freed device memory" << std::endl;
}

void CUDAGameOfLife::initializeGrid(const std::vector<bool>& initial_state)
{
    if (initial_state.empty())
    {
        initialRandomGrid(0.3);
    }
    else
    {
        if (initial_state.size() != getTotalCells())
        {
            std::cerr << "Initial state size does not match total cells" << std::endl;
            exit(1);
        }
        copyToDevice(initial_state);
    }
}

void CUDAGameOfLife::initialRandomGrid(float density)
{
    if (density < 0.0f || density > 1.0f)
    {
        std::cerr << "Density must be between 0.0 and 1.0" << std::endl;
        exit(1);
    }
    initialKernel<<<grid_size, block_size>>>(
        d_grid_current,
        params.width,
        params.height, 
        params.depth,
        density
    );
    cudaDeviceSynchronize();
}

void CUDAGameOfLife::initialCenterGrid(float density, int center_size)
{
    if (density < 0.0f || density > 1.0f)
    {
        std::cerr << "Density must be between 0.0 and 1.0" << std::endl;
        exit(1);
    }
    int min_dim = std::min(params.width, std::min(params.height, params.depth));
    if (center_size < 1 || center_size > min_dim / 2)
    {
        std::cerr << "Center size must be between 1 and half the smallest dimension" << std::endl;
        exit(1);
    }
    centerInitialKernel<<<grid_size, block_size>>>(
        d_grid_current,
        params.width,
        params.height, 
        params.depth,
        density,
        center_size
    );
    cudaDeviceSynchronize();
}
    
void CUDAGameOfLife::setGameParameters(const GameOfLifeParams& new_params)
{
    bool dimensions_changed = (params.width != new_params.width || 
        params.height != new_params.height || 
        params.depth != new_params.depth);
    params = new_params;

    if (dimensions_changed)
    {
        freeDeviceMemory();
        calculateExecutionConfig();
        allocateDeviceMemory();
    }
}
    
void CUDAGameOfLife::evolveGeneration()
{
    evolveKernel<<<grid_size, block_size>>>(
        d_grid_current, 
        d_grid_next, 
        params.width, 
        params.height, 
        params.depth,
        params.birth_min, 
        params.birth_max,
        params.survival_min, 
        params.survival_max
    );
    cudaDeviceSynchronize();
    std::swap(d_grid_current, d_grid_next);
}
    
void CUDAGameOfLife::copyToDevice(const std::vector<bool>& h_data)
{
    // copy from host to device - create temporary array
    size_t byte_size = h_data.size() * sizeof(bool);
    bool* temp_data = new bool[h_data.size()];
    for (size_t i = 0; i < h_data.size(); i++) 
    {
        temp_data[i] = h_data[i];
    }
    
    cudaError_t err = cudaMemcpy(d_grid_current, temp_data, byte_size, cudaMemcpyHostToDevice);
    delete[] temp_data;
    
    if (err != cudaSuccess) 
    {
        std::cerr << "copyToDevice failed: " << cudaGetErrorString(err) << std::endl;
    }
}
    
void CUDAGameOfLife::copyToHost(std::vector<bool>& h_data) const
{
    // resize host vector to match device data size
    h_data.resize(getTotalCells());
    
    // copy from device to host - create temporary array
    size_t byte_size = getTotalCells() * sizeof(bool);
    bool* temp_data = new bool[getTotalCells()];
    
    cudaError_t err = cudaMemcpy(temp_data, d_grid_current, byte_size, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
        for (int i = 0; i < getTotalCells(); i++) 
        {
            h_data[i] = temp_data[i];
        }
    }
    delete[] temp_data;
    
    if (err != cudaSuccess) 
    {
        std::cerr << "copyToHost failed: " << cudaGetErrorString(err) << std::endl;
    }
}

