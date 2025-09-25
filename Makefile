CXX = g++
NVCC = nvcc

CXXFLAGS = -std=c++20 -Iinclude
NVCCFLAGS = -std=c++20 -Iinclude

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
