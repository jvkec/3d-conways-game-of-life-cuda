#!/bin/bash

# CUDA Game of Life Batch Simulation Script
# This script automates your workflow: simulate -> save -> push -> notify

set -e  # Exit on any error

# Configuration
GRID_SIZE="96x96x96"
GENERATIONS=1000
SAVE_INTERVAL=10
OUTPUT_DIR="simulation_states"
PREFIX="sim"
DENSITY=0.3
RULES="14,19,14,19"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== CUDA Game of Life Batch Simulation ===${NC}"
echo -e "${YELLOW}Grid: ${GRID_SIZE}${NC}"
echo -e "${YELLOW}Generations: ${GENERATIONS}${NC}"
echo -e "${YELLOW}Save every: ${SAVE_INTERVAL} generations${NC}"
echo -e "${YELLOW}Output: ${OUTPUT_DIR}/${NC}"
echo ""

# Step 1: Build the batch runner
echo -e "${BLUE}Building batch runner...${NC}"
make batch_runner
if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi
echo -e "${GREEN}Build successful!${NC}"

# Step 2: Run simulation
echo -e "${BLUE}Starting simulation...${NC}"
./batch_runner \
    --grid ${GRID_SIZE} \
    --generations ${GENERATIONS} \
    --save-every ${SAVE_INTERVAL} \
    --output ${OUTPUT_DIR} \
    --prefix ${PREFIX} \
    --density ${DENSITY} \
    --rules ${RULES} \
    --verbose

if [ $? -ne 0 ]; then
    echo -e "${RED}Simulation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Simulation completed successfully!${NC}"

# Step 3: Generate transfer info
echo -e "${BLUE}Generating transfer information...${NC}"
cat > ${OUTPUT_DIR}/transfer_info.txt << EOF
CUDA Simulation Complete
========================

Grid: ${GRID_SIZE}
Generations: ${GENERATIONS}
Save Interval: ${SAVE_INTERVAL}
Files Created: $(ls ${OUTPUT_DIR}/*.bin | wc -l)
Total Size: $(du -sh ${OUTPUT_DIR} | cut -f1)

Ready for transfer to Mac for OpenGL rendering!

Next steps on Mac:
1. Pull this repository
2. Load states in OpenGL renderer
3. Playback saved frames

Files to transfer:
$(ls ${OUTPUT_DIR}/*.bin | head -10)
$(if [ $(ls ${OUTPUT_DIR}/*.bin | wc -l) -gt 10 ]; then echo "... and $(($(ls ${OUTPUT_DIR}/*.bin | wc -l) - 10)) more files"; fi)
EOF

echo -e "${GREEN}Transfer info saved to ${OUTPUT_DIR}/transfer_info.txt${NC}"

# Step 4: Git operations
echo -e "${BLUE}Preparing for git push...${NC}"

# Add new files
git add ${OUTPUT_DIR}/
git add src/cuda/state_manager.cu src/cuda/state_manager.h
git add src/batch_runner.cpp
git add run_simulation.sh

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo -e "${YELLOW}No changes to commit${NC}"
else
    echo -e "${BLUE}Committing changes...${NC}"
    git commit -m "Add batch simulation system and state management

- Added StateManager for efficient binary state serialization
- Added batch_runner for command-line batch processing  
- Added run_simulation.sh for automated workflow
- Generated ${GENERATIONS} generations with ${SAVE_INTERVAL} save interval
- Grid size: ${GRID_SIZE}, Density: ${DENSITY}"
    
    echo -e "${BLUE}Pushing to repository...${NC}"
    git push
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully pushed to repository!${NC}"
    else
        echo -e "${RED}Git push failed!${NC}"
        exit 1
    fi
fi

# Step 5: Summary
echo ""
echo -e "${GREEN}=== Batch Simulation Complete ===${NC}"
echo -e "${YELLOW}States saved: ${OUTPUT_DIR}/${NC}"
echo -e "${YELLOW}Files created: $(ls ${OUTPUT_DIR}/*.bin | wc -l)${NC}"
echo -e "${YELLOW}Total size: $(du -sh ${OUTPUT_DIR} | cut -f1)${NC}"
echo ""
