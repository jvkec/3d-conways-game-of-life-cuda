#pragma once

#include <GL/glew.h>        // extension manager
#include <GLFW/glfw3.h>     // window manager

class OpenGLManager
{
public:
    OpenGLManager();
    ~OpenGLManager();

    void init(int width = 800, int height = 600);
    void run();
    void cleanup();
    
    // accessors (const methods)
    GLFWwindow* getWindow() const { return window; };    
    bool shouldClose() const;

private:
    GLFWwindow* window;
    int windowWidth;
    int windowHeight;    

    // call backs (trigger function on events)
    // https://www.glfw.org/docs/3.3/input_guide.html#input_key
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);             // when user resizes the window
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);   // when user presses a key
    static void mouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset);        // when user scrolls the mouse
    
    // *we use static methods because we want to be able to call them without an instance of the class*
};