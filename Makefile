NVCC = nvcc
NVCC_FLAGS = --allow-unsupported-compiler -std=c++17 -O2
CXX = g++
CXX_FLAGS = -std=c++17 -O2
INCLUDES = -I./include -I/opt/homebrew/include
LIBRARY_PATHS = -L/opt/homebrew/lib
OPENGL_LIBS = -lglfw -lGLEW -framework OpenGL
CUDA_SOURCES = src/cuda/game_logic.cu src/cuda/kernels.cu src/cuda/state_manager.cu
OPENGL_SOURCES = src/opengl/opengl_manager.cpp src/opengl/renderer_3d.cpp src/opengl/camera.cpp
TEST_SOURCE = src/test_cuda.cpp
TEST_3D_SOURCE = src/test_3d_rules.cpp
BATCH_RUNNER_SOURCE = src/batch_runner.cpp

# cuda library
libcuda_game_logic.a: $(CUDA_SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -lib -o $@ $(CUDA_SOURCES)

test_cuda: libcuda_game_logic.a $(TEST_SOURCE)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $(TEST_SOURCE) -L. -lcuda_game_logic

test_3d_rules: libcuda_game_logic.a $(TEST_3D_SOURCE)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $(TEST_3D_SOURCE) -L. -lcuda_game_logic

batch_runner: libcuda_game_logic.a $(BATCH_RUNNER_SOURCE)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $(BATCH_RUNNER_SOURCE) -L. -lcuda_game_logic

# opengl 3d viewer application
viewer_3d: $(OPENGL_SOURCES) src/main.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $(LIBRARY_PATHS) -o $@ src/main.cpp $(OPENGL_SOURCES) $(OPENGL_LIBS)

# combined cuda + opengl application (for future integration)
main_app: libcuda_game_logic.a src/main.cpp $(OPENGL_SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBRARY_PATHS) -o $@ src/main.cpp $(OPENGL_SOURCES) -L. -lcuda_game_logic $(OPENGL_LIBS)

clean:
	rm -f *.a *.o test_cuda test_3d_rules main_app batch_runner viewer_3d

# default target
all: viewer_3d

# test target
test: test_cuda

.PHONY: clean all
