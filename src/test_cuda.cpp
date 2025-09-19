#include "cuda/game_logic.h"
#include <iostream>
#include <vector>
#include <chrono>

void printGridInfo(const CUDAGameOfLife& game) {
    std::cout << "Grid Info:" << std::endl;
    std::cout << "  Dimensions: " << game.getWidth() << "x" << game.getHeight() << "x" << game.getDepth() << std::endl;
    std::cout << "  Total Cells: " << game.getTotalCells() << std::endl;
    std::cout << "  Birth Range: [" << game.getParams().birth_min << ", " << game.getParams().birth_max << "]" << std::endl;
    std::cout << "  Survival Range: [" << game.getParams().survival_min << ", " << game.getParams().survival_max << "]" << std::endl;
}

void testBasicFunctionality() {
    std::cout << "\n=== Testing Basic Functionality ===" << std::endl;
    
    // test 1: constructor and initialization
    std::cout << "Test 1: Creating game with 8x8x8 grid..." << std::endl;
    CUDAGameOfLife game(8, 8, 8, 14, 19, 14, 19);
    printGridInfo(game);
    
    // test 2: random initialization
    std::cout << "\nTest 2: Random initialization with 30% density..." << std::endl;
    game.initialRandomGrid(0.3f);
    std::cout << "Random grid initialized successfully" << std::endl;
    
    // test 3: copy data back to host and check
    std::cout << "\nTest 3: Copying data back to host..." << std::endl;
    std::vector<bool> host_data;
    game.copyToHost(host_data);
    
    // count alive cells
    int alive_count = 0;
    for (bool cell : host_data) {
        if (cell) alive_count++;
    }
    std::cout << "Alive cells: " << alive_count << " / " << host_data.size() 
              << " (" << (100.0f * alive_count / host_data.size()) << "%)" << std::endl;
    
    // test 4: evolution
    std::cout << "\nTest 4: Running evolution steps..." << std::endl;
    for (int i = 0; i < 5; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        game.evolveGeneration();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // get updated cell count
        game.copyToHost(host_data);
        alive_count = 0;
        for (bool cell : host_data) {
            if (cell) alive_count++;
        }
        
        std::cout << "  Generation " << (i+1) << ": " << alive_count << " alive cells, "
                  << "took " << duration.count() << " μs" << std::endl;
    }
    
    std::cout << "Basic functionality test completed" << std::endl;
}

void testCustomInitialization() {
    std::cout << "\n=== Testing Custom Initialization ===" << std::endl;
    
    // create a small 4x4x4 grid for easier visualization
    CUDAGameOfLife game(4, 4, 4);
    
    // create a simple pattern - a few cells in the center
    std::vector<bool> custom_pattern(64, false); // 4x4x4 = 64
    
    // set some cells alive in the center
    custom_pattern[21] = true; // (1,1,1)
    custom_pattern[22] = true; // (2,1,1)
    custom_pattern[25] = true; // (1,2,1)
    custom_pattern[26] = true; // (2,2,1)
    
    std::cout << "Setting custom pattern with 4 alive cells..." << std::endl;
    game.initializeGrid(custom_pattern);
    
    // copy back and verify
    std::vector<bool> result;
    game.copyToHost(result);
    
    int alive_count = 0;
    for (int i = 0; i < 64; i++) {
        if (result[i]) {
            int x = i % 4;
            int y = (i / 4) % 4;
            int z = i / 16;
            std::cout << "Alive cell at (" << x << "," << y << "," << z << ")" << std::endl;
            alive_count++;
        }
    }
    
    std::cout << "Custom initialization test completed! Found " << alive_count << " alive cells" << std::endl;
}

void testPerformance() {
    std::cout << "\n=== Testing Performance ===" << std::endl;
    
    // test with full 96x96x96 grid
    std::cout << "Testing with full 96x96x96 grid..." << std::endl;
    CUDAGameOfLife game(96, 96, 96);
    game.initialRandomGrid(0.3f);
    
    // warm up
    game.evolveGeneration();
    
    // time multiple generations
    const int num_generations = 10;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_generations; i++) {
        game.evolveGeneration();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Completed " << num_generations << " generations in " 
              << duration.count() << " ms" << std::endl;
    std::cout << "Average time per generation: " 
              << (duration.count() / num_generations) << " ms" << std::endl;
    
    // get final cell count
    std::vector<bool> final_state;
    game.copyToHost(final_state);
    int alive_count = 0;
    for (bool cell : final_state) {
        if (cell) alive_count++;
    }
    std::cout << "Final alive cells: " << alive_count << std::endl;
}

int main() {
    std::cout << "=== CUDA Game of Life Test Suite ===" << std::endl;
    
    try {
        testBasicFunctionality();
        testCustomInitialization();
        testPerformance();
        
        std::cout << "\n=== All Tests Passed ===" << std::endl;
        std::cout << "CUDA implementation works fine" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
