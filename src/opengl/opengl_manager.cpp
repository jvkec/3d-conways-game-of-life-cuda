#include "opengl/opengl_manager.h"
#include <iostream>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

OpenGLManager::OpenGLManager()
{
    // initialize to non-garbage to avoid undefined behavior
    window = nullptr;
    windowWidth = 0;
    windowHeight = 0;
    
    renderer3D = nullptr;
    camera = nullptr;
    
    firstMouse = true;
    lastX = 400.0f;
    lastY = 300.0f;
    deltaTime = 0.0f;
    lastFrame = 0.0f;
    animationPlaying = false;
    animationSpeed = 2.0f; // frames per second
    lastAnimationTime = 0.0f;
    
    for (int i = 0; i < 1024; i++) {
        keys[i] = false;
    }
}

OpenGLManager::~OpenGLManager()
{
    cleanup();
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

    // request opengl core profile
#ifdef __APPLE__
    // macOS supports up to 4.1 core; match shader #version 410
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#else
    // On other platforms, 3.3 core is widely supported
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#endif
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

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

    // initialize glew
    glewExperimental = GL_TRUE; // needed for core profile
    if (glewInit() != GLEW_OK)
    {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return;
    }

    // get actual framebuffer size (important for Retina displays on macOS)
    int framebufferWidth, framebufferHeight;
    glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
    
    // set initial viewport using actual framebuffer size
    glViewport(0, 0, framebufferWidth, framebufferHeight);
    
    // update stored dimensions to match actual framebuffer
    windowWidth = framebufferWidth;
    windowHeight = framebufferHeight;
    // log versions
    std::cout << "OpenGL Version:  " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL Version:    " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    // enable v-sync
    glfwSwapInterval(1);

    // set up call back functions
    // we pass function pointers to the callbacks so glfw can find the addresses of the functions
    // and execute them without an instance of the class when an event occurs
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetScrollCallback(window, mouseScrollCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    
    // store this instance in the window user pointer for callbacks
    glfwSetWindowUserPointer(window, this);
    
    // capture the mouse cursor
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    
    // initialize 3D components
    camera = new Camera(glm::vec3(0.0f, 0.0f, 150.0f));
    camera->setTarget(glm::vec3(0.0f));
    camera->setRadius(180.0f);
    renderer3D = new Renderer3D();
    
    if (!renderer3D->initialize()) {
        std::cerr << "Failed to initialize 3D renderer" << std::endl;
        delete renderer3D;
        renderer3D = nullptr;
        return;
    }
}

void OpenGLManager::run()
{
    while (!shouldClose())
    {
        // Calculate delta time
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        
        // Process input
        processInput();
        
        // Handle animation
        if (animationPlaying && renderer3D && renderer3D->getTotalFrames() > 0) {
            if (currentFrame - lastAnimationTime >= (1.0f / animationSpeed)) {
                renderer3D->nextFrame();
                lastAnimationTime = currentFrame;
            }
        }
        
        render(window);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cleanup();
}

void OpenGLManager::cleanup()
{
    if (renderer3D) {
        delete renderer3D;
        renderer3D = nullptr;
    }
    
    if (camera) {
        delete camera;
        camera = nullptr;
    }
    
    if (window) {
        glfwDestroyWindow(window);  // destroy the window
        window = nullptr;
    }
    
    glfwTerminate();            // terminate GLFW entirely
}

void OpenGLManager::render(GLFWwindow* window)
{
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    if (renderer3D && camera) {
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 
                                              (float)windowWidth / (float)windowHeight, 
                                              0.1f, 2000.0f);
        
        glm::mat4 view = camera->getViewMatrix();
        
        renderer3D->render(view, projection);
    }
}

bool OpenGLManager::shouldClose() const
{
    return glfwWindowShouldClose(window);
}

// static member function definitions
void OpenGLManager::framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
  
    OpenGLManager* manager = static_cast<OpenGLManager*>(glfwGetWindowUserPointer(window));
    if (manager) {
        manager->windowWidth = width;
        manager->windowHeight = height;
    }
}

void OpenGLManager::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    OpenGLManager* manager = static_cast<OpenGLManager*>(glfwGetWindowUserPointer(window));
    
    if (action == GLFW_PRESS) {
        if (key >= 0 && key < 1024) {
            manager->keys[key] = true;
        }
    } else if (action == GLFW_RELEASE) {
        if (key >= 0 && key < 1024) {
            manager->keys[key] = false;
        }
    }
    
    // Handle special keys
    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            case GLFW_KEY_SPACE:
                manager->animationPlaying = !manager->animationPlaying;
                std::cout << "Animation " << (manager->animationPlaying ? "playing" : "paused") << std::endl;
                break;
            case GLFW_KEY_RIGHT:
                if (manager->renderer3D) {
                    manager->renderer3D->nextFrame();
                    std::cout << "Frame: " << manager->renderer3D->getCurrentFrame() + 1 
                             << "/" << manager->renderer3D->getTotalFrames() << std::endl;
                }
                break;
            case GLFW_KEY_LEFT:
                if (manager->renderer3D) {
                    manager->renderer3D->prevFrame();
                    std::cout << "Frame: " << manager->renderer3D->getCurrentFrame() + 1 
                             << "/" << manager->renderer3D->getTotalFrames() << std::endl;
                }
                break;
            case GLFW_KEY_R:
                if (manager->camera) {
                    manager->camera->reset();
                    std::cout << "Camera reset" << std::endl;
                }
                break;
        }
    }
}

void OpenGLManager::mouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    OpenGLManager* manager = static_cast<OpenGLManager*>(glfwGetWindowUserPointer(window));
    if (manager->camera) {
        manager->camera->processMouseScroll(yoffset);
    }
}

void OpenGLManager::mouseCallback(GLFWwindow* window, double xpos, double ypos)
{
    OpenGLManager* manager = static_cast<OpenGLManager*>(glfwGetWindowUserPointer(window));
    
    if (manager->firstMouse) {
        manager->lastX = xpos;
        manager->lastY = ypos;
        manager->firstMouse = false;
    }
    
    float xoffset = xpos - manager->lastX;
    float yoffset = manager->lastY - ypos; // reversed since y-coordinates go from bottom to top
    
    manager->lastX = xpos;
    manager->lastY = ypos;
    
    if (manager->camera) {
        manager->camera->processMouseMovement(xoffset, yoffset);
    }
}

void OpenGLManager::processInput()
{
    // Orbit-only mode: no keyboard camera movement
    (void)deltaTime;
}

void OpenGLManager::loadSimulationData(const std::string& directory)
{
    if (renderer3D) {
        renderer3D->loadAllStates(directory);
        std::cout << "Loaded simulation data from: " << directory << std::endl;
        std::cout << "Controls:" << std::endl;
        std::cout << "  Mouse: Orbit (rotate) around grid" << std::endl;
        std::cout << "  Space: Play/Pause animation" << std::endl;
        std::cout << "  Left/Right arrows: Previous/Next frame" << std::endl;
        std::cout << "  R: Reset camera" << std::endl;
        std::cout << "  Escape: Exit" << std::endl;
    }
}