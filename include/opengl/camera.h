#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

enum class CameraMovement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

class Camera {
public:
    // cam attributes
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 worldUp;
    
    // orbit attributes
    float yaw;
    float pitch;
    
    // orbit options
    glm::vec3 target;   // point to orbit around
    float radius;       // fixed
    float mouseSensitivity;
    
    Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 150.0f),
           glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),
           float yaw = -90.0f,
           float pitch = 0.0f);
    
    // returns the view matrix calculated using Euler angles and the LookAt Matrix
    glm::mat4 getViewMatrix();
    
    // keyboard input is ignored (no-op) in orbit mode
    void processKeyboard(CameraMovement direction, float deltaTime);
    
    // mouse input is used to orbit the camera around the target
    void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = true);
    
    // mouse scroll is ignored (no-op) in orbit mode
    void processMouseScroll(float yoffset);
    
    void reset();

    // orit controls
    void setTarget(const glm::vec3& t) { target = t; updateCameraVectors(); }
    void setRadius(float r) { radius = r; updateCameraVectors(); }
    float getRadius() const { return radius; }

private:
    void updateCameraVectors();
};
