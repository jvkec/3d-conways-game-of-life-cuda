#include "cuda/game_logic.h"
#include "cuda/state_manager.h"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <filesystem>
#include <ctime>

/**
 * BatchRunner - Command-line tool for running batch simulations
 * 
 * This tool automates your workflow:
 * 1. Run simulations on GPU machine
 * 2. Save states periodically 
 * 3. Generate metadata for easy transfer to Mac
 * 
 * Usage examples:
 *   ./batch_runner --grid 96x96x96 --generations 1000 --save-every 10 --output states/
 *   ./batch_runner --load state_000100.bin --continue 500 --save-every 5
 *   ./batch_runner --list states/ --validate
 */

struct BatchConfig {
    int width = 96;
    int height = 96;
    int depth = 96;
    int birth_min = 14;
    int birth_max = 19;
    int survival_min = 14;
    int survival_max = 19;
    float density = 0.3f;
    uint32_t num_generations = 100;
    uint32_t save_interval = 1;
    std::string output_dir = "runs/states";
    std::string prefix = "state";
    std::string load_file = "";
    bool validate_only = false;
    bool list_only = false;
    bool verbose = false;
    bool center_init = false;
    int center_size = 15;
};

void printUsage(const char* program_name) {
    std::cout << "CUDA Game of Life Batch Runner\n";
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --grid WxHxD          Grid dimensions (default: 96x96x96)\n";
    std::cout << "  --generations N       Number of generations to simulate (default: 100)\n";
    std::cout << "  --save-every N        Save state every N generations (default: 1)\n";
    std::cout << "  --output DIR          Output directory (default: states)\n";
    std::cout << "  --prefix PREFIX       Filename prefix (default: state)\n";
    std::cout << "  --density FLOAT       Initial population density (default: 0.3)\n";
    std::cout << "  --rules B_MIN,B_MAX,S_MIN,S_MAX  Game rules (default: 14,19,14,19)\n";
    std::cout << "  --load FILE           Load initial state from file\n";
    std::cout << "  --continue N          Continue simulation for N more generations\n";
    std::cout << "  --list DIR            List all state files in directory\n";
    std::cout << "  --validate            Validate state files\n";
    std::cout << "  --center-init         Initialize with pattern in center\n";
    std::cout << "  --center-size N       Size of center region (default: 15)\n";
    std::cout << "  --verbose             Verbose output\n";
    std::cout << "  --help                Show this help\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " --grid 64x64x64 --generations 500 --save-every 10\n";
    std::cout << "  " << program_name << " --load state_000100.bin --continue 200\n";
    std::cout << "  " << program_name << " --list states/ --validate\n";
}

BatchConfig parseArguments(int argc, char* argv[]) {
    BatchConfig config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printUsage(argv[0]);
            exit(0);
        }
        else if (arg == "--grid" && i + 1 < argc) {
            std::string grid_str = argv[++i];
            size_t x_pos = grid_str.find('x');
            size_t y_pos = grid_str.find('x', x_pos + 1);
            
            if (x_pos != std::string::npos && y_pos != std::string::npos) {
                config.width = std::stoi(grid_str.substr(0, x_pos));
                config.height = std::stoi(grid_str.substr(x_pos + 1, y_pos - x_pos - 1));
                config.depth = std::stoi(grid_str.substr(y_pos + 1));
            }
        }
        else if (arg == "--generations" && i + 1 < argc) {
            config.num_generations = std::stoul(argv[++i]);
        }
        else if (arg == "--save-every" && i + 1 < argc) {
            config.save_interval = std::stoul(argv[++i]);
        }
        else if (arg == "--output" && i + 1 < argc) {
            config.output_dir = argv[++i];
        }
        else if (arg == "--prefix" && i + 1 < argc) {
            config.prefix = argv[++i];
        }
        else if (arg == "--density" && i + 1 < argc) {
            config.density = std::stof(argv[++i]);
        }
        else if (arg == "--rules" && i + 1 < argc) {
            std::string rules_str = argv[++i];
            std::vector<int> rules;
            size_t start = 0;
            size_t end = rules_str.find(',');
            
            while (end != std::string::npos) {
                rules.push_back(std::stoi(rules_str.substr(start, end - start)));
                start = end + 1;
                end = rules_str.find(',', start);
            }
            rules.push_back(std::stoi(rules_str.substr(start)));
            
            if (rules.size() == 4) {
                config.birth_min = rules[0];
                config.birth_max = rules[1];
                config.survival_min = rules[2];
                config.survival_max = rules[3];
            }
        }
        else if (arg == "--load" && i + 1 < argc) {
            config.load_file = argv[++i];
        }
        else if (arg == "--continue" && i + 1 < argc) {
            config.num_generations = std::stoul(argv[++i]);
        }
        else if (arg == "--list" && i + 1 < argc) {
            config.output_dir = argv[++i];
            config.list_only = true;
        }
        else if (arg == "--validate") {
            config.validate_only = true;
        }
        else if (arg == "--center-init") {
            config.center_init = true;
        }
        else if (arg == "--center-size" && i + 1 < argc) {
            config.center_size = std::stoi(argv[++i]);
        }
        else if (arg == "--verbose") {
            config.verbose = true;
        }
    }
    
    return config;
}

void listStates(const std::string& directory, bool validate) {
    std::vector<std::pair<std::string, StateManager::StateHeader>> states;
    int count = StateManager::listStates(directory, states);
    
    if (count == 0) {
        std::cout << "No state files found in " << directory << std::endl;
        return;
    }
    
    std::cout << "Found " << count << " state files:\n\n";
    std::cout << "Generation | Dimensions | Rules        | Timestamp           | Valid\n";
    std::cout << "-----------|------------|--------------|---------------------|-------\n";
    
    for (const auto& state : states) {
        const auto& header = state.second;
        std::string filename = state.first.substr(state.first.find_last_of("/\\") + 1);
        
        // Format timestamp
        std::time_t time_t = header.timestamp;
        std::string time_str = std::ctime(&time_t);
        time_str.pop_back(); // Remove newline
        
        bool valid = true;
        if (validate) {
            valid = StateManager::validateStateFile(state.first);
        }
        
        printf("%-10u | %dx%dx%d | %d,%d,%d,%d | %s | %s\n",
               header.generation,
               header.width, header.height, header.depth,
               header.birth_min, header.birth_max, 
               header.survival_min, header.survival_max,
               time_str.c_str(),
               valid ? "Yes" : "No");
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== CUDA Game of Life Batch Runner ===" << std::endl;
    
    BatchConfig config = parseArguments(argc, argv);
    
    if (config.list_only) {
        listStates(config.output_dir, config.validate_only);
        return 0;
    }
    
    try {
        CUDAGameOfLife game;
        
        if (!config.load_file.empty()) {
            // Load existing state
            std::cout << "Loading state from " << config.load_file << "..." << std::endl;
            uint32_t start_generation;
            if (!StateManager::loadState(config.load_file, game, start_generation)) {
                std::cerr << "Failed to load state file!" << std::endl;
                return 1;
            }
            std::cout << "Starting from generation " << start_generation << std::endl;
        } else {
            // Create new game
            std::cout << "Creating new game with " << config.width << "x" << config.height 
                      << "x" << config.depth << " grid..." << std::endl;
            
            GameOfLifeParams params;
            params.width = config.width;
            params.height = config.height;
            params.depth = config.depth;
            params.birth_min = config.birth_min;
            params.birth_max = config.birth_max;
            params.survival_min = config.survival_min;
            params.survival_max = config.survival_max;
            
            game.setGameParameters(params);
            
            if (config.center_init) {
                game.initialCenterGrid(config.density, config.center_size);
                std::cout << "Initialized with " << (config.density * 100) << "% density in center region (size " 
                          << config.center_size << ")" << std::endl;
            } else {
                game.initialRandomGrid(config.density);
                std::cout << "Initialized with " << (config.density * 100) << "% density" << std::endl;
            }
        }
        
        // Run batch simulation
        std::cout << "\nStarting batch simulation..." << std::endl;
        std::cout << "Generations: " << config.num_generations << std::endl;
        std::cout << "Save interval: " << config.save_interval << std::endl;
        std::cout << "Output directory: " << config.output_dir << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int saved_count = StateManager::runBatchSimulation(
            game, config.output_dir, config.num_generations, 
            config.save_interval, config.prefix
        );
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "\n=== Batch Simulation Complete ===" << std::endl;
        std::cout << "Total time: " << duration.count() << " ms" << std::endl;
        std::cout << "States saved: " << saved_count << std::endl;
        std::cout << "Average time per generation: " 
                  << (duration.count() / config.num_generations) << " ms" << std::endl;
        
        // Generate summary file for easy transfer
        std::string summary_file = config.output_dir + "/simulation_summary.txt";
        std::ofstream summary(summary_file);
        if (summary.is_open()) {
            summary << "CUDA Game of Life Simulation Summary\n";
            summary << "=====================================\n\n";
            summary << "Grid dimensions: " << config.width << "x" << config.height 
                    << "x" << config.depth << "\n";
            summary << "Total generations: " << config.num_generations << "\n";
            summary << "Save interval: " << config.save_interval << "\n";
            summary << "States saved: " << saved_count << "\n";
            summary << "Total time: " << duration.count() << " ms\n";
            summary << "Rules: birth[" << config.birth_min << "," << config.birth_max 
                    << "] survival[" << config.survival_min << "," << config.survival_max << "]\n";
            summary << "Initial density: " << (config.density * 100) << "%\n\n";
            summary << "Files created:\n";
            
            for (int i = 0; i <= config.num_generations; i += config.save_interval) {
                char filename[256];
                snprintf(filename, sizeof(filename), "%s_%06u.bin", config.prefix.c_str(), i);
                summary << "  " << filename << "\n";
            }
            
            summary.close();
            std::cout << "Summary saved to: " << summary_file << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
