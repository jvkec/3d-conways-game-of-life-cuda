#ifndef GAME_LOGIC_H    // use header guards as they are safer than #pragma once
#define GAME_LOGIC_H

#include <cuda_runtime.h>
#include <types.h>
#include <vector>
 
class CUDAGameOfLife 
{
private:
    // device memory pointers
    // d_ prefix for device (gpu) and h_ for host (cpu) is best practice
    bool* d_grid_current;   // curr generation on gpu
    bool* d_grid_next;      // next generation on gpu

    int grid_width;
    int grid_height;
    int grid_depth;
    int total_cells;

    // cuda extension configs
    dim3 block_size;
    dim3 grid_size;

    GameOfLifeParameters params;

    // helpers
    void calculateExecutionConfig();
    void allocateDeviceMemory();
    void freeDeviceMemory();

public:
    CUDAGameOfLife(int width, int height, int depth);
    ~CUDAGameOfLife();

    void initializeGrid(const std::vector<bool>& initial_state = {});
    void initialRandomGrid(float density = 0.3);

    // game
    void setGameParameters(const GameOfLifeParams& new_params);
    void evolveGeneration();    // runs one generation

    // data copying between host and device
    void copyToDevice(const std::vector<bool>& h_data);
    void copyToHost(const std::vector<bool>& d_data);

    // accessors
    int getTotalCells() const { return total_cells; }
    int getWidth() const { return grid_width; }
    int getHeight() const { return grid_height; }
    int getDepth() const { return grid_depth; }
};

#endif  // GAME_LOGIC_H