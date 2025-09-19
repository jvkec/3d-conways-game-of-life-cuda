#include "opengl/renderer_3d.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cstdint>

Renderer3D::Renderer3D() : shaderProgram(0), VAO(0), VBO(0), EBO(0), instanceVBO(0) {}

Renderer3D::~Renderer3D() {
    cleanup();
}

bool Renderer3D::initialize() {
    if (!createShaders()) {
        std::cerr << "Failed to create shaders" << std::endl;
        return false;
    }
    
    createCubeGeometry();
    setupBuffers();
    
    // enable depth testing
    glEnable(GL_DEPTH_TEST);
    
    // disable face culling to improve visibility with transparency
    glDisable(GL_CULL_FACE);

    // enable alpha blending to see through dense regions
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    return true;
}

void Renderer3D::cleanup() {
    if (shaderProgram) glDeleteProgram(shaderProgram);
    if (VAO) glDeleteVertexArrays(1, &VAO);
    if (VBO) glDeleteBuffers(1, &VBO);
    if (EBO) glDeleteBuffers(1, &EBO);
    if (instanceVBO) glDeleteBuffers(1, &instanceVBO);
}

bool Renderer3D::loadSimulationState(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return false;
    }
    
    // helper: build alive cells from boolean grid
    auto buildAliveFromGrid = [&](const std::vector<bool>& grid, int w, int h, int d) {
        currentAliveCells.clear();
        gridWidth = w; gridHeight = h; gridDepth = d;
        size_t total = static_cast<size_t>(w) * static_cast<size_t>(h) * static_cast<size_t>(d);
        currentAliveCells.reserve( static_cast<size_t>(total / 8) ); // rough reserve for sparse
        for (size_t i = 0; i < total; ++i) {
            if (grid[i]) {
                int x = static_cast<int>(i % w);
                int y = static_cast<int>((i / w) % h);
                int z = static_cast<int>(i / (static_cast<size_t>(w) * h));
                currentAliveCells.emplace_back(x, y, z);
            }
        }
    };
    
    // try to detect CGOL header format written by StateManager
    const uint32_t MAGIC = 0x43474F4C; // "CGOL"
    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
    if (!file.good()) { file.close(); return false; }
    
    if (magic == MAGIC) {
        // read the rest of the header fields individually (avoid packing concerns)
        uint32_t version = 0;
        uint32_t w = 0, h = 0, d = 0;
        uint32_t generation = 0;
        uint64_t timestamp = 0;
        uint32_t birth_min = 0, birth_max = 0, survival_min = 0, survival_max = 0;
        uint32_t data_size = 0;
        uint32_t checksum = 0;
        file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&w), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&h), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&generation), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&timestamp), sizeof(uint64_t));
        file.read(reinterpret_cast<char*>(&birth_min), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&birth_max), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&survival_min), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&survival_max), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&data_size), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&checksum), sizeof(uint32_t));
        
        // read compressed data
        std::vector<uint8_t> compressed(data_size);
        if (data_size > 0) {
            file.read(reinterpret_cast<char*>(compressed.data()), data_size);
        }
        file.close();
        
        // decompress (RLE pairs: [run_length:uint8, value:uint8])
        size_t total_cells = static_cast<size_t>(w) * static_cast<size_t>(h) * static_cast<size_t>(d);
        std::vector<bool> grid;
        grid.reserve(total_cells);
        for (size_t i = 0; i + 1 < compressed.size(); i += 2) {
            uint8_t run = compressed[i];
            bool value = compressed[i + 1] != 0;
            for (uint8_t r = 0; r < run; ++r) {
                grid.push_back(value);
            }
        }
        grid.resize(total_cells, false);
        
        buildAliveFromGrid(grid, static_cast<int>(w), static_cast<int>(h), static_cast<int>(d));
    } else {
        // legacy formats: either [count][triplets...] or just [triplets...]
        file.seekg(0, std::ios::beg);
        currentAliveCells.clear();
        struct Cell { int32_t x, y, z; };
        // detect if file starts with count header (size % 12 == 4)
        file.seekg(0, std::ios::end);
        std::streamoff fsize = file.tellg();
        file.seekg(0, std::ios::beg);
        if (fsize >= 4 && (fsize % static_cast<std::streamoff>(sizeof(Cell))) == static_cast<std::streamoff>(sizeof(int32_t))) {
            int32_t count = 0;
            file.read(reinterpret_cast<char*>(&count), sizeof(int32_t));
            currentAliveCells.reserve(std::max(0, count));
            for (int32_t i = 0; i < count; ++i) {
                Cell cell;
                if (!file.read(reinterpret_cast<char*>(&cell), sizeof(Cell))) break;
                currentAliveCells.emplace_back(cell.x, cell.y, cell.z);
            }
        } else {
            Cell cell;
            while (file.read(reinterpret_cast<char*>(&cell), sizeof(Cell))) {
                currentAliveCells.emplace_back(cell.x, cell.y, cell.z);
            }
        }
        file.close();
    }
    
    updateInstanceData();
    return true;
}

void Renderer3D::loadAllStates(const std::string& directory) {
    framesAliveCells.clear();
    
    // find all .bin files in the directory
    std::vector<std::string> binFiles;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().extension() == ".bin") {
            binFiles.push_back(entry.path().string());
        }
    }
    
    // sort files by name to ensure correct order
    std::sort(binFiles.begin(), binFiles.end());
    
    // load each state
    for (const auto& filepath : binFiles) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) continue;
        
        std::vector<glm::ivec3> frame;
        const uint32_t MAGIC = 0x43474F4C; // "CGOL"
        uint32_t magic = 0;
        file.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
        if (file.good() && magic == MAGIC) {
            uint32_t version = 0;
            uint32_t w = 0, h = 0, d = 0;
            uint32_t generation = 0;
            uint64_t timestamp = 0;
            uint32_t birth_min = 0, birth_max = 0, survival_min = 0, survival_max = 0;
            uint32_t data_size = 0;
            uint32_t checksum = 0;
            file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&w), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&h), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&generation), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&timestamp), sizeof(uint64_t));
            file.read(reinterpret_cast<char*>(&birth_min), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&birth_max), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&survival_min), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&survival_max), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&data_size), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&checksum), sizeof(uint32_t));
            
            std::vector<uint8_t> compressed(data_size);
            if (data_size > 0) {
                file.read(reinterpret_cast<char*>(compressed.data()), data_size);
            }
            file.close();
            
            size_t total_cells = static_cast<size_t>(w) * static_cast<size_t>(h) * static_cast<size_t>(d);
            std::vector<bool> grid;
            grid.reserve(total_cells);
            for (size_t i = 0; i + 1 < compressed.size(); i += 2) {
                uint8_t run = compressed[i];
                bool value = compressed[i + 1] != 0;
                for (uint8_t r = 0; r < run; ++r) grid.push_back(value);
            }
            grid.resize(total_cells, false);
            
            // set grid dims from the first file encountered
            if (framesAliveCells.empty()) {
                gridWidth = static_cast<int>(w);
                gridHeight = static_cast<int>(h);
                gridDepth = static_cast<int>(d);
            }
            
            // build frame alive coords
            frame.reserve(total_cells / 8);
            for (size_t i = 0; i < total_cells; ++i) {
                if (grid[i]) {
                    int x = static_cast<int>(i % w);
                    int y = static_cast<int>((i / w) % h);
                    int z = static_cast<int>(i / (static_cast<size_t>(w) * h));
                    frame.emplace_back(x, y, z);
                }
            }
        } else {
            // legacy
            file.seekg(0, std::ios::beg);
            struct Cell { int32_t x, y, z; };
            file.seekg(0, std::ios::end);
            std::streamoff fsize = file.tellg();
            file.seekg(0, std::ios::beg);
            if (fsize >= 4 && (fsize % static_cast<std::streamoff>(sizeof(Cell))) == static_cast<std::streamoff>(sizeof(int32_t))) {
                int32_t count = 0;
                file.read(reinterpret_cast<char*>(&count), sizeof(int32_t));
                frame.reserve(std::max(0, count));
                for (int32_t i = 0; i < count; ++i) {
                    Cell cell;
                    if (!file.read(reinterpret_cast<char*>(&cell), sizeof(Cell))) break;
                    frame.emplace_back(cell.x, cell.y, cell.z);
                }
            } else {
                Cell cell;
                while (file.read(reinterpret_cast<char*>(&cell), sizeof(Cell))) {
                    frame.emplace_back(cell.x, cell.y, cell.z);
                }
            }
            file.close();
        }
        
        framesAliveCells.push_back(std::move(frame));
        std::cout << "Loaded: " << filepath << std::endl;
    }
    
    if (!framesAliveCells.empty()) {
        currentFrame = 0;
        currentAliveCells = framesAliveCells[0];
        updateInstanceData();
    }
    
    std::cout << "Loaded " << framesAliveCells.size() << " simulation states" << std::endl;
}

void Renderer3D::nextFrame() {
    if (!framesAliveCells.empty()) {
        currentFrame = (currentFrame + 1) % framesAliveCells.size();
        currentAliveCells = framesAliveCells[currentFrame];
        updateInstanceData();
    }
}

void Renderer3D::prevFrame() {
    if (!framesAliveCells.empty()) {
        currentFrame = (currentFrame - 1 + static_cast<int>(framesAliveCells.size())) % static_cast<int>(framesAliveCells.size());
        currentAliveCells = framesAliveCells[currentFrame];
        updateInstanceData();
    }
}

void Renderer3D::setFrame(int frame) {
    if (!framesAliveCells.empty() && frame >= 0 && frame < static_cast<int>(framesAliveCells.size())) {
        currentFrame = frame;
        currentAliveCells = framesAliveCells[currentFrame];
        updateInstanceData();
    }
}

void Renderer3D::render(const glm::mat4& view, const glm::mat4& projection) {
    glUseProgram(shaderProgram);
    
    // Set matrices
    glm::mat4 model = glm::mat4(1.0f);
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
    
    // Set color with alpha so dense cubes aren't a solid block
    glm::vec4 color(0.2f, 0.8f, 0.3f, 0.15f);
    glUniform4fv(colorLoc, 1, glm::value_ptr(color));
    
    // Render instances
    glBindVertexArray(VAO);
    glDrawElementsInstanced(GL_TRIANGLES, static_cast<GLsizei>(cubeIndices.size()), GL_UNSIGNED_INT, 0, static_cast<GLsizei>(instancePositions.size()));
    glBindVertexArray(0);
}

bool Renderer3D::createShaders() {
    shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    if (shaderProgram == 0) {
        return false;
    }
    
    // Get uniform locations
    modelLoc = glGetUniformLocation(shaderProgram, "model");
    viewLoc = glGetUniformLocation(shaderProgram, "view");
    projectionLoc = glGetUniformLocation(shaderProgram, "projection");
    colorLoc = glGetUniformLocation(shaderProgram, "color");
    
    return true;
}

void Renderer3D::createCubeGeometry() {
    // cube vertices (positions only)
    cubeVertices = {
        // front face
        -0.35f, -0.35f,  0.35f,
         0.35f, -0.35f,  0.35f,
         0.35f,  0.35f,  0.35f,
        -0.35f,  0.35f,  0.35f,
        
        // back face
        -0.35f, -0.35f, -0.35f,
         0.35f, -0.35f, -0.35f,
         0.35f,  0.35f, -0.35f,
        -0.35f,  0.35f, -0.35f
    };
    
    // cube indices
    cubeIndices = {
        // front face
        0, 1, 2, 2, 3, 0,
        // back face
        4, 5, 6, 6, 7, 4,
        // left face
        7, 3, 0, 0, 4, 7,
        // right face
        1, 5, 6, 6, 2, 1,
        // top face
        3, 2, 6, 6, 7, 3,
        // bottom face
        0, 1, 5, 5, 4, 0
    };
}

void Renderer3D::setupBuffers() {
    // generate and bind VAO
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    
    // vertex buffer
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, cubeVertices.size() * sizeof(float), cubeVertices.data(), GL_STATIC_DRAW);
    
    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // element buffer
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, cubeIndices.size() * sizeof(unsigned int), cubeIndices.data(), GL_STATIC_DRAW);
    
    // instance buffer
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    
    // instance position attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1); // Tell OpenGL this is an instanced vertex attribute
    
    glBindVertexArray(0);
}

void Renderer3D::updateInstanceData() {
    instancePositions.clear();
    
    // convert alive cell coordinates to world positions and center the grid
    float centerOffsetX = (gridWidth - 1) * 0.5f;
    float centerOffsetY = (gridHeight - 1) * 0.5f;
    float centerOffsetZ = (gridDepth - 1) * 0.5f;
    for (const auto& c : currentAliveCells) {
        instancePositions.emplace_back(
            static_cast<float>(c.x) - centerOffsetX,
            static_cast<float>(c.y) - centerOffsetY,
            static_cast<float>(c.z) - centerOffsetZ
        );
    }
    
    // update instance buffer
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, instancePositions.size() * sizeof(glm::vec3), instancePositions.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    std::cout << "Updated instance data: " << instancePositions.size() << " alive cells" << std::endl;
}

GLuint Renderer3D::compileShader(const std::string& source, GLenum type) {
    GLuint shader = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    
    // check compilation status
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation failed: " << infoLog << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    
    return shader;
}

GLuint Renderer3D::createShaderProgram(const std::string& vertexSource, const std::string& fragmentSource) {
    GLuint vertexShader = compileShader(vertexSource, GL_VERTEX_SHADER);
    GLuint fragmentShader = compileShader(fragmentSource, GL_FRAGMENT_SHADER);
    
    if (vertexShader == 0 || fragmentShader == 0) {
        return 0;
    }
    
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    
    // check linking status
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
        glDeleteProgram(program);
        program = 0;
    }
    
    // clean up shaders
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return program;
}
