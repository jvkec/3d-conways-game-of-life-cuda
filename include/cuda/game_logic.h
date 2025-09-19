#ifndef GAME_LOGIC_H    // use header guards as they are safer than #pragma once
#define GAME_LOGIC_H

#include <cuda_runtime.h>
#include "cuda/types.h"
#include <vector>
 
class CUDAGameOfLife 
{
private:
    // device memory pointers
    // use double buffer for curr and next gens
    // d_ prefix for device (gpu) and h_ for host (cpu) is best practice
    bool* d_grid_current;
    bool* d_grid_next;
    
    // game parameters
    GameOfLifeParams params;

    // cuda extension configs
    dim3 block_size;
    dim3 grid_size;

    // helpers
    void calculateExecutionConfig();
    void allocateDeviceMemory();
    void freeDeviceMemory();

public:
    // single constructor with default parameters
    CUDAGameOfLife(int width = 96, int height = 96, int depth = 96, 
                   int birth_min = 14, int birth_max = 19, 
                   int survival_min = 14, int survival_max = 19);
    CUDAGameOfLife(const GameOfLifeParams& game_params);
    
    ~CUDAGameOfLife();

    void initializeGrid(const std::vector<bool>& initial_state = {});
    void initialRandomGrid(float density = 0.3);

    // game
    void setGameParameters(const GameOfLifeParams& new_params);
    void evolveGeneration();    // runs one generation

    // data copying between host and device
    void copyToDevice(const std::vector<bool>& h_data);
    void copyToHost(std::vector<bool>& d_data) const;

    // accessors
    int getTotalCells() const { return params.width * params.height * params.depth; }
    int getWidth() const { return params.width; }
    int getHeight() const { return params.height; }
    int getDepth() const { return params.depth; }
    const GameOfLifeParams& getParams() const { return params; }
};

#endif  // GAME_LOGIC_H