#include <stdio.h>

__global__ void hello_world() {
    printf("Hello, World!\n");
}

int main() {
    hello_world<<<1, 1>>>();
    cudaDeviceSynchronize(); // ensures all gpu operations are completed before the cpu continues to terminate the program
    return 0;
}