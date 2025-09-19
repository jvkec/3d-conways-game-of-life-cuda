#ifndef STATE_MANAGER_H
#define STATE_MANAGER_H

#include "cuda/game_logic.h"
#include <string>
#include <vector>
#include <fstream>
#include <chrono>

/**
 * StateManager - Handles serialization and deserialization of game states
 * 
 * This class provides efficient binary storage for Conway's Game of Life states,
 * optimized for your batch workflow between GPU machine and Mac rendering.
 * 
 * Key features:
 * - Binary format for fast I/O
 * - Metadata tracking (generation, timestamp, parameters)
 * - Compression support for large grids
 * - Validation and error handling
 */
class StateManager {
public:
    struct __attribute__((packed)) StateHeader 
    {
        uint32_t magic_number;      // File format identifier
        uint32_t version;           // Format version
        uint32_t width, height, depth;
        uint32_t generation;
        uint64_t timestamp;         // Unix timestamp
        uint32_t birth_min, birth_max;
        uint32_t survival_min, survival_max;
        uint32_t data_size;         // Size of compressed data
        uint32_t checksum;          // Simple checksum for validation
    };
    
    static constexpr uint32_t MAGIC_NUMBER = 0x43474F4C; // "CGOL" in hex
    static constexpr uint32_t CURRENT_VERSION = 1;
    
    /**
     * Save a game state to a binary file
     * @param game The CUDA game instance to save
     * @param filename Output filename
     * @param generation Current generation number
     * @return true if successful, false otherwise
     */
    static bool saveState(const CUDAGameOfLife& game, 
                         const std::string& filename, 
                         uint32_t generation = 0);
    
    /**
     * Load a game state from a binary file
     * @param filename Input filename
     * @param game CUDA game instance to populate
     * @param generation Output parameter for generation number
     * @return true if successful, false otherwise
     */
    static bool loadState(const std::string& filename, 
                         CUDAGameOfLife& game, 
                         uint32_t& generation);
    
    /**
     * Get metadata from a state file without loading the full state
     * @param filename Input filename
     * @param header Output header structure
     * @return true if successful, false otherwise
     */
    static bool getStateMetadata(const std::string& filename, StateHeader& header);
    
    /**
     * Validate a state file integrity
     * @param filename Input filename
     * @return true if valid, false otherwise
     */
    static bool validateStateFile(const std::string& filename);
    
    /**
     * Create a batch of state files from a simulation
     * @param game Initial game state
     * @param output_dir Directory to save states
     * @param num_generations Total generations to simulate
     * @param save_interval How often to save (every N generations)
     * @param prefix Filename prefix for saved states
     * @return Number of states successfully saved
     */
    static int runBatchSimulation(CUDAGameOfLife& game,
                                 const std::string& output_dir,
                                 uint32_t num_generations,
                                 uint32_t save_interval = 1,
                                 const std::string& prefix = "state");
    
    /**
     * List all state files in a directory with metadata
     * @param directory Directory to scan
     * @param states Output vector of state information
     * @return Number of state files found
     */
    static int listStates(const std::string& directory, 
                         std::vector<std::pair<std::string, StateHeader>>& states);

private:
    /**
     * Calculate simple checksum for data validation
     */
    static uint32_t calculateChecksum(const std::vector<bool>& data);
    
    /**
     * Compress boolean data using simple run-length encoding
     * This helps reduce file sizes for sparse populations
     */
    static std::vector<uint8_t> compressData(const std::vector<bool>& data);
    
    /**
     * Decompress data back to boolean vector
     */
    static std::vector<bool> decompressData(const std::vector<uint8_t>& compressed, 
                                           size_t original_size);
};

#endif // STATE_MANAGER_H
