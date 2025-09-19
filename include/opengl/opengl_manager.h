#pragma once

#include <GL/glew.h>        // extension manager
#include <GLFW/glfw3.h>     // window manager
#include "opengl/renderer_3d.h"
#include "opengl/camera.h"
#include <string>

class OpenGLManager
{
public:
    OpenGLManager();
    ~OpenGLManager();

    void init(int width = 800, int height = 600);
    void run();
    void cleanup();
    void render(GLFWwindow* window);
    
    void loadSimulationData(const std::string& directory);
    
    // accessors (const methods)
    GLFWwindow* getWindow() const { return window; };    
    bool shouldClose() const;   // checks if program should close

private:
    GLFWwindow* window;
    int windowWidth;
    int windowHeight;
    
    Renderer3D* renderer3D;
    Camera* camera;
    
    bool firstMouse;
    float lastX, lastY;
    float deltaTime, lastFrame;
    bool keys[1024];
    bool animationPlaying;
    float animationSpeed;
    float lastAnimationTime;

    // call backs (trigger function on events)
    // https://www.glfw.org/docs/3.3/input_guide.html#input_key
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);             // when user resizes the window
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);   // when user presses a key
    static void mouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset);        // when user scrolls the mouse
    static void mouseCallback(GLFWwindow* window, double xpos, double ypos);                    // when user moves the mouse
    
    // Input processing
    void processInput();
    
    // *we use static methods because we want to be able to call them without an instance of the class*
};