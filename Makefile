#!/usr/bin/make -f

# Target and directories
SRCDIR := src
BUILDDIR := build
TGTDIR := bin
TARGET := spmv

# Compilers and flags
CC := gcc
CPPC := g++
NVCC := nvcc
CUDA_KERNEL_CHECK_FLAG ?= -DCUDA_CHECK_KERNELS=1
MYGPU_ARCH ?= sm_70

# Prepare files
SOURCES_C := $(shell find $(SRCDIR) -type f -name *.c)
OBJECTS_C := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES_C:.c=.o))
SOURCES_CPP := $(shell find $(SRCDIR) -type f -name *.cpp)
OBJECTS_CPP := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES_CPP:.cpp=.o))
SOURCES_CU := $(shell find $(SRCDIR) -type f -name *.cu)
OBJECTS_CU := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES_CU:.cu=.o))
SOURCES := $(SOURCES_C) $(SOURCES_CPP) $(SOURCES_CU)
OBJECTS := $(OBJECTS_C) $(OBJECTS_CPP) $(OBJECTS_CU)

# Flags
CCFLAGS := -O3 -fopenmp -std=c99 -Wno-unused-result $(JACOBI_ITERS_FLAG) $(CUDA_KERNEL_CHECK_FLAG) # -g 
CPPCFLAGS := -O3 -fopenmp --std=c++11 -Wno-unused-result -fno-exceptions -Wall -Wextra $(JACOBI_ITERS_FLAG) $(CUDA_KERNEL_CHECK_FLAG) # -g
NVCCFLAGS := -ccbin g++ -O3 -Xcompiler -fopenmp -Xcompiler -Wno-unused-result -Xcompiler -fno-exceptions -Xcompiler -Wall $(JACOBI_ITERS_FLAG) $(CUDA_KERNEL_CHECK_FLAG) # -Xcompiler -g -g -G
ARCHFLAGS := -arch=$(MYGPU_ARCH) -Wno-deprecated-gpu-targets
LDFLAGS := -lrt -lm -lcudart -fopenmp -lhsl_mc64 -lgfortran
# NOTE: Be careful with the order of libraries above
# NOTE: -g option for valgrind to track lines 

# Directories
LIB := -Llib -L$(CUDA_PATH)/lib64 -L/usr/local/lib -L$(HOME)/lib
INC := -Iinclude -I$(CUDA_PATH)/include -Itemplates

# First rule
all: $(TGTDIR)/$(TARGET) | $(TGTDIR)

# Linking
$(TGTDIR)/$(TARGET): $(OBJECTS) | $(BUILDDIR)
	$(CC) $^ -o $(TGTDIR)/$(TARGET) $(INC) $(LDFLAGS) $(LIB)

# C compilations
$(BUILDDIR)/%.o: $(SRCDIR)/%.c include/*.h 
	$(CC) $(CCFLAGS) $(INC) -c -o $@ $<

# CPP compilations
$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp include/*.hpp templates/*.tpp
	$(CPPC) $(CPPCFLAGS) $(INC) -c -o $@ $<

# CUDA compilations
$(BUILDDIR)/%.o: $(SRCDIR)/%.cu include/*.cuh templates/*.tpp
	$(NVCC) $(NVCCFLAGS) $(ARCHFLAGS) $(INC) -c -o $@ $<

# Objects directory
$(BUILDDIR):
	@mkdir -p $(BUILDDIR)  

# Target directory
$(TGTDIR):
	@mkdir -p $(TGTDIR)  

# Cleaning
clean: 
	$(RM) -r $(BUILDDIR)/*.o

# Diagnostic
show:
	@echo "Sources: $(SOURCES)"
	@echo "Objects: $(OBJECTS)"
	@echo "CUDA HOME: $(CUDA_PATH)"
	@echo "Target arch: $(MYGPU_ARCH)"

# Code distribution overall
cloc:
	cloc .

# Clean and make again
again:
	@make clean && make

.PHONY: all clean show cloc again
