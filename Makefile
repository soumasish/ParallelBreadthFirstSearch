TARGET = bfs

SRCDIR = src
OBJDIR = obj
BINDIR = bin
INClDIR = includes

CUDA_INSTALL_PATH := /Developer/NVIDIA/CUDA-7.5
GCC := $(CUDA_INSTALL_PATH)/bin/nvcc
LIBS := -I. -I$(SRCDIR) -I$(CUDA_INSTALL_PATH)/include -I$(INClDIR)
CUDA_LIBS := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart

SOURCES := $(wildcard $(SRCDIR)/*.cu)
INCLUDES := $(wildcard $(INClDIR)/*.h)
OBJECTS := $(SOURCES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)
rm = rm -f

$(BINDIR)/$(TARGET) : $(OBJECTS)
	mkdir -p $(BINDIR)
	$(GCC) -o $@  $(OBJECTS)
	@echo "Linking complete!"

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cu
	@$(GCC) $(LIBS) -c $(SRCDIR)/*.cu -odir $(OBJDIR)
	@echo "Compiled "$<" successfully!"

.PHONEY: clean
clean:
	@$(rm)   $(OBJECTS)
	@echo "Cleanup complete!"
remove: clean
	@$(rm) $(BINDIR)/$(TARGET)
	@echo "Executable removed!"
