CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -g -O0 -Wall
CUDAFLAGS = -std=c++17 -arch=sm_75 -g -G -O0
INCLUDES = -I. -I./common -I./ops -I./tensor -I./utils
LIBS = -lcudart
TARGET = main
BUILD_DIR = build

# 源文件（去掉 test 目录）
CUDA_SRCS = main.cu tensor/tensor.cu ops/scalarops.cu ops/multiple.cu
CPP_SRCS = utils/rand.cpp

# 目标文件（放在 build 目录）
CUDA_OBJS = $(patsubst %.cu,$(BUILD_DIR)/%.o,$(CUDA_SRCS))
CPP_OBJS = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))
OBJS = $(CUDA_OBJS) $(CPP_OBJS)

# 默认目标
all: $(BUILD_DIR)/$(TARGET)

# 创建构建目录结构
$(shell mkdir -p $(BUILD_DIR))
$(shell mkdir -p $(BUILD_DIR)/tensor)
$(shell mkdir -p $(BUILD_DIR)/ops)
$(shell mkdir -p $(BUILD_DIR)/utils)

# 链接
$(BUILD_DIR)/$(TARGET): $(OBJS)
	$(NVCC) $(CUDAFLAGS) $(INCLUDES) $^ -o $@ $(LIBS)

# CUDA 文件编译（放在 build 目录）
$(BUILD_DIR)/%.o: %.cu
	$(NVCC) $(CUDAFLAGS) $(INCLUDES) -c $< -o $@

# C++ 文件编译（放在 build 目录）
$(BUILD_DIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# 清理
clean:
	rm -rf $(BUILD_DIR)

# 运行
run: $(BUILD_DIR)/$(TARGET)
	./$(BUILD_DIR)/$(TARGET)

# 调试
debug: $(BUILD_DIR)/$(TARGET)
	cuda-gdb ./$(BUILD_DIR)/$(TARGET)

# 显示变量（调试用）
print-%:
	@echo $* = $($*)

.PHONY: all clean run debug