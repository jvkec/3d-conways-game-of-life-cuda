#include "opengl_manager.h"
#include <iostream>

OpenGLManager::OpenGLManager()
{
    // initialize to non-garbage to avoid undefined behavior
    window = nullptr;
    windowWidth = 0;
    windowHeight = 0;
}

OpenGLManager::~OpenGLManager()
{
    // initialize to non-garbage to avoid undefined behavior
    window = nullptr;
    windowWidth = 0;
    windowHeight = 0;
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
    while (!shouldClose())
    {
        render(window);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cleanup();
}

void OpenGLManager::cleanup()
{
    glfwDestroyWindow(window);  // destroy the window
    glfwTerminate();            // terminate GLFW entirely
}

void OpenGLManager::render(GLFWwindow* window)
{
    // GL_COLOR_BUFFER_BIT inducates buffers currently enabled for color writing
    glClear(GL_COLOR_BUFFER_BIT);
}

bool OpenGLManager::shouldClose() const
{
    return glfwWindowShouldClose(window);
}

// no need for this to be a member function because we want to be able 
// to call it without an instance of the class
static void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    // escape key to close window
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

static void mouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    // scroll up to move camera down, scroll down to move camera up
    // scroll left to move camera right, scroll right to move camera left
    // cameraPos is a vec3 that stores the camera's position
    // we can access the x, y, and z components of the cameraPos vector
    // using cameraPos.x, cameraPos.y, and cameraPos.z
    // we can also access the x, y, and z components of the cameraPos vector
    // using cameraPos.x, cameraPos.y, and cameraPos.z
    // if (yoffset > 0)
    // {
    //     cameraPos.y -= 0.1;
    // }
    // else if (yoffset < 0)
    // {
    //     cameraPos.y += 0.1;
    // }
    // else if (xoffset > 0)
    // {
    //     cameraPos.x -= 0.1;
    // }
    // else if (xoffset < 0)
    // {
    //     cameraPos.x += 0.1;
    //}
}