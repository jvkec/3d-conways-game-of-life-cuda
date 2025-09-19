#include "opengl/camera.h"

Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch)
    : front(glm::vec3(0.0f, 0.0f, -1.0f))
    , mouseSensitivity(0.1f) {
    this->position = position;
    this->worldUp = up;
    this->yaw = yaw;
    this->pitch = pitch;
    this->target = glm::vec3(0.0f);
    this->radius = glm::length(position - target);
    updateCameraVectors();
}

glm::mat4 Camera::getViewMatrix() {
    return glm::lookAt(position, target, up);
}

void Camera::processKeyboard(CameraMovement direction, float deltaTime) {
    (void)direction;
    (void)deltaTime;
}

void Camera::processMouseMovement(float xoffset, float yoffset, bool constrainPitch) {
    xoffset *= mouseSensitivity;
    yoffset *= mouseSensitivity;
    
    yaw += xoffset;
    pitch += yoffset;
    
    // make sure that when pitch is out of bounds, screen doesn't get flipped
    if (constrainPitch) {
        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;
    }
    
    updateCameraVectors();
}

void Camera::processMouseScroll(float yoffset) {
    // no zoom in orbit-fixed mode
    (void)yoffset;
}

void Camera::reset() {
    target = glm::vec3(0.0f);
    radius = 150.0f;
    yaw = -90.0f;
    pitch = 0.0f;
    updateCameraVectors();
}

void Camera::updateCameraVectors() {
    // compute camera position from spherical coordinates around target
    float radYaw = glm::radians(yaw);
    float radPitch = glm::radians(pitch);
    glm::vec3 offset;
    offset.x = radius * cos(radPitch) * cos(radYaw);
    offset.y = radius * sin(radPitch);
    offset.z = radius * cos(radPitch) * sin(radYaw);
    position = target + offset;

    // front vector looks at target
    front = glm::normalize(target - position);
    // right and up vectors
    right = glm::normalize(glm::cross(front, worldUp));
    up = glm::normalize(glm::cross(right, front));
}
