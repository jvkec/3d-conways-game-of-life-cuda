#include "opengl_manager.h"
#include <iostream>

OpenGLManager::OpenGLManager()
{
}

OpenGLManager::~OpenGLManager()
{
}

void OpenGLManager::init(int width, int height)
{
    // initialize GLFW
    windowWidth = width;
    windowHeight = height;

    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return;
    }

    // create a window
    window = glfwCreateWindow(windowWidth, windowHeight, "OpenGL Manager", NULL, NULL);
    if (!window)
    {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        return;
    }

    // make context current
    glfwMakeContextCurrent(window);

    // initialize GLEW
    if (glewInit() != GLEW_OK)
    {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return;
    }

    // set up call back functions
    // we pass function pointers to the callbacks so glfw can find the addresses of the functions
    // and execute them without an instance of the class when an event occurs
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetScrollCallback(window, mouseScrollCallback);
}

void OpenGLManager::run()
{

}

void OpenGLManager::cleanup()
{
    
}

bool OpenGLManager::shouldClose() const
{
    return glfwWindowShouldClose(window);
}

// no need for this to be a member function because we want to be able 
// to call it without an instance of the class
static void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{

}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{

}

static void mouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{

}