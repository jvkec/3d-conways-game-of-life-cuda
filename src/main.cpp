#include "opengl/opengl_manager.h"
#include <iostream>

int main()
{

    std::cout << "Starting OpenGL Manager" << std::endl;
    // loop to create and destroy the window
    // this is to avoid the window from being destroyed
    // when the program is closed

    OpenGLManager openglManager;
    openglManager.init();
    openglManager.run();
    openglManager.cleanup();
}