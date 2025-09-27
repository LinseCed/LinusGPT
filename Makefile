CXX = g++
NVCC = nvcc

MODE ?= release

ifeq ($(MODE), debug)
	CXXFLAGS = -std=c++20 -Iinclude -g -O0 -Wall -Wextra
	NVCCFLAGS = -std=c++20 -Iinclude -G -g -O0
else ifeq ($(MODE), release)
	CXXFLAGS = -std=c++20 -Iinclude -O3 -march=native -s
	NVCCFLAGS = -std=c++20 -Iinclude -O3 -Xcompiler "-s -O3"
endif

SRC_CPP := $(shell find src -name "*.cpp")
SRC_CU  := $(shell find src -name "*.cu")

OBJ_CPP := $(SRC_CPP:.cpp=.o)
OBJ_CU  := $(SRC_CU:.cu=.o)

TARGET = llm

all: $(TARGET)

$(TARGET): $(OBJ_CPP) $(OBJ_CU)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ_CPP) $(OBJ_CU) $(TARGET)
