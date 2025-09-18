#include <game_logic.h>

// "default" constructor
CUDAGameOfLife::CUDAGameOfLife(int width, int height, int depth, 
                               int birth_min, int birth_max, 
                               int survival_min, int survival_max)
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

}

void CUDAGameOfLife::allocateDeviceMemory()
{

}

void CUDAGameOfLife::freeDeviceMemory()
{


}

void CUDAGameOfLife::initializeGrid(const std::vector<bool>& initial_state)
{

}

void CUDAGameOfLife::initialRandomGrid(float density)
{

}
    
void CUDAGameOfLife::setGameParameters(const GameOfLifeParams& new_params)
{

}
    
void CUDAGameOfLife::evolveGeneration()
{

}
    
void CUDAGameOfLife::copyToDevice(const std::vector<bool>& h_data)
{
    // copy from host to device
    cudaMemcpy(d_grid_current, h_data.data(), 
               h_data.size() * sizeof(bool), 
               cudaMemcpyHostToDevice);
}
    
void CUDAGameOfLife::copyToHost(std::vector<bool>& h_data)
{
    // resize host vector to match device data size
    h_data.resize(getTotalCells());
    
    // copy from device to host
    cudaMemcpy(h_data.data(), d_grid_current, 
               getTotalCells() * sizeof(bool), 
               cudaMemcpyDeviceToHost);
}

