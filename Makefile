NVCC = nvcc
NVCC_FLAGS = --allow-unsupported-compiler -std=c++17 -O2
INCLUDES = -I./include
CUDA_SOURCES = src/cuda/game_logic.cu src/cuda/kernels.cu
TEST_SOURCE = src/test_cuda.cpp

# cuda library
libcuda_game_logic.a: $(CUDA_SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -lib -o $@ $(CUDA_SOURCES)

# Build the test program
test_cuda: libcuda_game_logic.a $(TEST_SOURCE)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $(TEST_SOURCE) -L. -lcuda_game_logic

# for later
main_app: libcuda_game_logic.a src/main.cpp src/opengl/opengl_manager.cpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ src/main.cpp src/opengl/opengl_manager.cpp -L. -lcuda_game_logic -lGL -lGLU

clean:
	rm -f *.a *.o test_cuda main_app

# default target for test
all: test_cuda

.PHONY: clean all
