#include "opengl/opengl_manager.h"
#include <iostream>
#include <filesystem>

int main(int argc, char** argv)
{
    std::cout << "Starting 3D Conway's Game of Life Viewer" << std::endl;
    
    OpenGLManager openglManager;
    openglManager.init(1200, 800);
    
    // loading simulation data
    std::string dir = (argc > 1) ? argv[1] : "";
    if (dir.empty()) {
        if (std::filesystem::exists("massive_growth")) dir = "massive_growth";
        else dir = "states";
    }
    openglManager.loadSimulationData(dir);
    
    std::cout << "Starting render loop..." << std::endl;
    openglManager.run();
    
    return 0;
}