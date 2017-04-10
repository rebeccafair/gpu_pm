################################################################################
#
## Build script for project
#
#################################################################################

# Add source files here
EXECUTABLE      := patternTest

# # Cuda source files (compiled with nvcc)
CUFILES_sm_35   := gpu_test.cu main.cu

# # C/C++ source files (compiled with gcc / c++)

SRCDIR=./src/
CCFILES := eventReader.cpp patternReader.cpp matchPatterns.cpp
CXXFLAGS += -std=c++11 -O3

# # Compiler-specific flags
#NVCCFLAGS      := --ptxas-options="-v" -std=c++11
#NVCCFLAGS := -lineinfo # For getting line number in cuda-memcheck
#NVCCFLAGS := -O3
# ################################################################################
# # Rules and targets

include common.mk

