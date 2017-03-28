################################################################################
#
## Build script for project
#
#################################################################################

# Add source files here
EXECUTABLE      := patternTest

# # Cuda source files (compiled with nvcc)
#CUFILES_sm_35   := KernelDriver.cu
CUFILES_sm_35   := main.cu

# # C/C++ source files (compiled with gcc / c++)

SRCDIR=./src/
CCFILES := eventReader.cpp patternReader.cpp matchPatterns.cpp
CXXFLAGS += -std=c++11

# # Compiler-specific flags
#NVCCFLAGS      := --ptxas-options="-v" -std=c++11

#NVCCFLAGS :=  --keep

# ################################################################################
# # Rules and targets

include common.mk

