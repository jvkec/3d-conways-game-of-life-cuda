#include "cuda/state_manager.h"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cstring>

bool StateManager::saveState(const CUDAGameOfLife& game, 
                            const std::string& filename, 
                            uint32_t generation) {
    try 
    {
        // copy current state from GPU to host
        std::vector<bool> host_data;
        game.copyToHost(host_data);
        
        StateHeader header;
        header.magic_number = MAGIC_NUMBER;
        header.version = CURRENT_VERSION;
        header.width = game.getWidth();
        header.height = game.getHeight();
        header.depth = game.getDepth();
        header.generation = generation;
        header.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        const auto& params = game.getParams();
        header.birth_min = params.birth_min;
        header.birth_max = params.birth_max;
        header.survival_min = params.survival_min;
        header.survival_max = params.survival_max;
        
        // compress data
        std::vector<uint8_t> compressed_data = compressData(host_data);
        header.data_size = compressed_data.size();
        header.checksum = calculateChecksum(host_data);
        
        // write to file
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return false;
        }
        
        // write header
        file.write(reinterpret_cast<const char*>(&header), sizeof(StateHeader));
        
        // write compressed data
        file.write(reinterpret_cast<const char*>(compressed_data.data()), compressed_data.size());
        
        file.close();
        
        std::cout << "Saved state to " << filename 
                  << " (generation " << generation 
                  << ", " << compressed_data.size() << " bytes)" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error saving state: " << e.what() << std::endl;
        return false;
    }
}

bool StateManager::loadState(const std::string& filename, 
                            CUDAGameOfLife& game, 
                            uint32_t& generation) {
    try {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for reading: " << filename << std::endl;
            return false;
        }
        
        // read header
        StateHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(StateHeader));
        
        // validate header
        if (header.magic_number != MAGIC_NUMBER) {
            std::cerr << "Invalid file format: " << filename << std::endl;
            return false;
        }
        
        if (header.version != CURRENT_VERSION) {
            std::cerr << "Unsupported file version: " << header.version << std::endl;
            return false;
        }
        
        // read compressed data
        std::vector<uint8_t> compressed_data(header.data_size);
        file.read(reinterpret_cast<char*>(compressed_data.data()), header.data_size);
        
        file.close();
        
        // decompress data
        size_t total_cells = header.width * header.height * header.depth;
        std::vector<bool> host_data = decompressData(compressed_data, total_cells);
        
        // validate checksum
        if (calculateChecksum(host_data) != header.checksum) {
            std::cerr << "Checksum validation failed: " << filename << std::endl;
            return false;
        }
        
        // create game with loaded parameters
        GameOfLifeParams params;
        params.width = header.width;
        params.height = header.height;
        params.depth = header.depth;
        params.birth_min = header.birth_min;
        params.birth_max = header.birth_max;
        params.survival_min = header.survival_min;
        params.survival_max = header.survival_max;
        
        game.setGameParameters(params);
        game.initializeGrid(host_data);
        
        generation = header.generation;
        
        std::cout << "Loaded state from " << filename 
                  << " (generation " << generation << ")" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading state: " << e.what() << std::endl;
        return false;
    }
}

bool StateManager::getStateMetadata(const std::string& filename, StateHeader& header) {
    try {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }
        
        file.read(reinterpret_cast<char*>(&header), sizeof(StateHeader));
        file.close();
        
        return (header.magic_number == MAGIC_NUMBER && header.version == CURRENT_VERSION);
        
    } catch (const std::exception& e) {
        std::cerr << "Error reading metadata: " << e.what() << std::endl;
        return false;
    }
}

bool StateManager::validateStateFile(const std::string& filename) {
    StateHeader header;
    if (!getStateMetadata(filename, header)) {
        return false;
    }
    
    // check if file size matches expected size
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return false;
    }
    
    size_t file_size = file.tellg();
    size_t expected_size = sizeof(StateHeader) + header.data_size;
    
    file.close();
    
    return (file_size == expected_size);
}

int StateManager::runBatchSimulation(CUDAGameOfLife& game,
                                    const std::string& output_dir,
                                    uint32_t num_generations,
                                    uint32_t save_interval,
                                    const std::string& prefix) {
    try {
        // create output directory if it doesn't exist
        std::filesystem::create_directories(output_dir);
        
        int saved_count = 0;
        
        // save initial state
        std::string filename = output_dir + "/" + prefix + "_000000.bin";
        if (saveState(game, filename, 0)) {
            saved_count++;
        }
        
        // run simulation
        for (uint32_t gen = 1; gen <= num_generations; gen++) {
            game.evolveGeneration();
            
            // save state at intervals
            if (gen % save_interval == 0) {
                char filename_buffer[256];
                snprintf(filename_buffer, sizeof(filename_buffer), 
                        "%s/%s_%06u.bin", output_dir.c_str(), prefix.c_str(), gen);
                
                if (saveState(game, filename_buffer, gen)) {
                    saved_count++;
                }
            }
            
            // progress indicator
            if (num_generations >= 10 && gen % (num_generations / 10) == 0) {
                std::cout << "Progress: " << (100 * gen / num_generations) << "%" << std::endl;
            }
        }
        
        std::cout << "Batch simulation completed. Saved " << saved_count << " states." << std::endl;
        return saved_count;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in batch simulation: " << e.what() << std::endl;
        return 0;
    }
}

int StateManager::listStates(const std::string& directory, 
                            std::vector<std::pair<std::string, StateHeader>>& states) {
    states.clear();
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file() && entry.path().extension() == ".bin") {
                StateHeader header;
                if (getStateMetadata(entry.path().string(), header)) {
                    states.emplace_back(entry.path().string(), header);
                }
            }
        }
        
        // sort by generation number
        std::sort(states.begin(), states.end(), 
                 [](const auto& a, const auto& b) {
                     return a.second.generation < b.second.generation;
                 });
        
        return states.size();
        
    } catch (const std::exception& e) {
        std::cerr << "Error listing states: " << e.what() << std::endl;
        return 0;
    }
}

uint32_t StateManager::calculateChecksum(const std::vector<bool>& data) {
    uint32_t checksum = 0;
    for (size_t i = 0; i < data.size(); i += 32) {
        uint32_t chunk = 0;
        for (int j = 0; j < 32 && (i + j) < data.size(); j++) {
            if (data[i + j]) {
                chunk |= (1U << j);
            }
        }
        checksum ^= chunk;
    }
    return checksum;
}

std::vector<uint8_t> StateManager::compressData(const std::vector<bool>& data) {
    // Simple run-length encoding for sparse data
    std::vector<uint8_t> compressed;
    
    bool current_value = false;
    uint32_t run_length = 0;
    
    for (bool bit : data) {
        if (bit == current_value) {
            run_length++;
            if (run_length == 255) { // max run length
                compressed.push_back(run_length);
                compressed.push_back(current_value ? 1 : 0);
                run_length = 0;
            }
        } else {
            if (run_length > 0) {
                compressed.push_back(run_length);
                compressed.push_back(current_value ? 1 : 0);
            }
            current_value = bit;
            run_length = 1;
        }
    }
    
    // add final run
    if (run_length > 0) {
        compressed.push_back(run_length);
        compressed.push_back(current_value ? 1 : 0);
    }
    
    return compressed;
}

std::vector<bool> StateManager::decompressData(const std::vector<uint8_t>& compressed, 
                                             size_t original_size) {
    std::vector<bool> data;
    data.reserve(original_size);
    
    for (size_t i = 0; i < compressed.size(); i += 2) {
        if (i + 1 >= compressed.size()) break;
        
        uint8_t run_length = compressed[i];
        bool value = compressed[i + 1] != 0;
        
        for (uint8_t j = 0; j < run_length; j++) {
            data.push_back(value);
        }
    }
    
    // ensure we have the correct size
    data.resize(original_size, false);
    
    return data;
}
