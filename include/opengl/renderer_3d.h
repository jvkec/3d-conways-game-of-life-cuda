#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <string>

class Renderer3D {
public:
    Renderer3D();
    ~Renderer3D();

    bool initialize();
    void cleanup();
    
    // load simulation state from .bin file
    bool loadSimulationState(const std::string& filepath);
    
    // render the current state
    void render(const glm::mat4& view, const glm::mat4& projection);
    
    // animation controls
    void loadAllStates(const std::string& directory);
    void nextFrame();
    void prevFrame();
    void setFrame(int frame);
    int getCurrentFrame() const { return currentFrame; }
    int getTotalFrames() const { return static_cast<int>(framesAliveCells.size()); }
    
    // grid properties
    int getGridWidth() const { return gridWidth; }
    int getGridHeight() const { return gridHeight; }
    int getGridDepth() const { return gridDepth; }

private:
    // grid dimensions
    int gridWidth = 96;
    int gridHeight = 96;
    int gridDepth = 96;
    
    // current simulation data
    std::vector<glm::ivec3> currentAliveCells;                 // alive cell coordinates per current frame
    std::vector<std::vector<glm::ivec3>> framesAliveCells;     // frames of alive cell coordinates
    int currentFrame = 0;
    
    // opengl objects
    GLuint shaderProgram;
    GLuint VAO, VBO, EBO;
    GLuint instanceVBO; // for instanced rendering
    
    // shader uniforms
    GLint modelLoc, viewLoc, projectionLoc;
    GLint colorLoc;
    
    // cube geometry
    std::vector<float> cubeVertices;
    std::vector<unsigned int> cubeIndices;
    std::vector<glm::vec3> instancePositions;
    
    // helper methods
    bool createShaders();
    void createCubeGeometry();
    void setupBuffers();
    void updateInstanceData();
    GLuint compileShader(const std::string& source, GLenum type);
    GLuint createShaderProgram(const std::string& vertexSource, const std::string& fragmentSource);
    
    // shader sources
    const std::string vertexShaderSource = R"(
        #version 410 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aInstancePos;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        out vec3 WorldPos;
        
        void main() {
            vec3 worldPos = aPos + aInstancePos;
            WorldPos = worldPos;
            gl_Position = projection * view * model * vec4(worldPos, 1.0);
        }
    )";
    
    const std::string fragmentShaderSource = R"(
        #version 410 core
        out vec4 FragColor;
        
        in vec3 WorldPos;
        
        uniform vec4 color;
        
        void main() {
            // Flat color for compatibility; add proper lighting later
            FragColor = color;
        }
    )";
};
