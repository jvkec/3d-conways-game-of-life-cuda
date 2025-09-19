#include "cuda/game_logic.h"
#include <iostream>
#include <vector>
#include <chrono>

void test3DRules() {
    std::cout << "=== Testing 3D Game of Life Rules ===" << std::endl;
    
    int birth_min = 5, birth_max = 8;   // 5-8 neighbors for birth
    int survival_min = 4, survival_max = 7;  // 4-7 neighbors for survival
    
    std::cout << "Testing with 3D-appropriate rules:" << std::endl;
    std::cout << "  Birth: " << birth_min << "-" << birth_max << " neighbors" << std::endl;
    std::cout << "  Survival: " << survival_min << "-" << survival_max << " neighbors" << std::endl;
    
    // create a small 6x6x6 grid for easier observation
    CUDAGameOfLife game(6, 6, 6, birth_min, birth_max, survival_min, survival_max);
    
    // initialize with 40% density
    game.initialRandomGrid(0.4f);
    
    std::vector<bool> host_data;
    game.copyToHost(host_data);
    
    int alive_count = 0;
    for (bool cell : host_data) {
        if (cell) alive_count++;
    }
    std::cout << "Initial alive cells: " << alive_count << " / " << host_data.size() 
              << " (" << (100.0f * alive_count / host_data.size()) << "%)" << std::endl;
    
    // run several generations and see how the population evolves
    std::cout << "\nEvolution over 10 generations:" << std::endl;
    for (int i = 0; i < 10; i++) {
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
        
        // stop if population stabilizes (no change for 3 generations)
        if (i >= 2 && alive_count == 0) {
            std::cout << "  Population died out after " << (i+1) << " generations." << std::endl;
            break;
        }
    }
    
    std::cout << "\n3D Game of Life test completed" << std::endl;
}

void testClassicRules() {
    std::cout << "\n=== Testing Classic 2D Rules in 3D ===" << std::endl;
    
    int birth_min = 6, birth_max = 6;
    int survival_min = 5, survival_max = 6;
    
    std::cout << "Testing with classic-inspired 3D rules:" << std::endl;
    std::cout << "  Birth: exactly " << birth_min << " neighbors" << std::endl;
    std::cout << "  Survival: " << survival_min << "-" << survival_max << " neighbors" << std::endl;
    
    CUDAGameOfLife game(8, 8, 8, birth_min, birth_max, survival_min, survival_max);
    game.initialRandomGrid(0.3f);
    
    std::vector<bool> host_data;
    game.copyToHost(host_data);
    
    int alive_count = 0;
    for (bool cell : host_data) {
        if (cell) alive_count++;
    }
    std::cout << "Initial alive cells: " << alive_count << " / " << host_data.size() 
              << " (" << (100.0f * alive_count / host_data.size()) << "%)" << std::endl;
    
    // run generations
    for (int i = 0; i < 8; i++) {
        game.evolveGeneration();
        game.copyToHost(host_data);
        alive_count = 0;
        for (bool cell : host_data) {
            if (cell) alive_count++;
        }
        std::cout << "  Generation " << (i+1) << ": " << alive_count << " alive cells" << std::endl;
        
        if (alive_count == 0) {
            std::cout << "  Population died out after " << (i+1) << " generations." << std::endl;
            break;
        }
    }
}

int main() {
    std::cout << "=== 3D Game of Life Rules Testing ===" << std::endl;
    
    try {
        test3DRules();
        testClassicRules();
        
        std::cout << "\n=== All 3D Rules Tests Completed ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
