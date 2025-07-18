################################################################################
# Makefile for MAX78000 EMG Classification System
# Based on Maxim Integrated SDK examples
################################################################################

# Project settings
PROJECT := emg_classification_max78000
TARGET := MAX78000
COMPILER := GCC
BOARD := EvKit_V1

# Source files
SRCS += emg_main.c
SRCS += emg_cnn.c
SRCS += emg_communication.c
SRCS += ../shared/protocols/mcu_communication.c

# Header paths
IPATH += ./src
IPATH += ../shared/protocols
IPATH += ../MAX78000_SDK/Libraries/Boards/MAX78000/Include
IPATH += ../MAX78000_SDK/Libraries/Boards/MAX78000/$(BOARD)
IPATH += ../MAX78000_SDK/Libraries/PeriphDrivers/Include/MAX78000
IPATH += ../MAX78000_SDK/Libraries/CMSIS/Device/Maxim/MAX78000/Include
IPATH += ../MAX78000_SDK/Libraries/CMSIS/Include

# Compiler flags
PROJ_CFLAGS += -DEMG_CNN_ACCELERATOR
PROJ_CFLAGS += -DMAX78000_SDK
PROJ_CFLAGS += -DARM_MATH_CM4
PROJ_CFLAGS += -D__FPU_PRESENT=1
PROJ_CFLAGS += -Wall
PROJ_CFLAGS += -Wextra
PROJ_CFLAGS += -Wno-unused-parameter
PROJ_CFLAGS += -Os
PROJ_CFLAGS += -ffunction-sections
PROJ_CFLAGS += -fdata-sections

# Linker flags
PROJ_LDFLAGS += -Wl,--gc-sections
PROJ_LDFLAGS += -Wl,--print-memory-usage

# Library paths
LIBPATH += ../MAX78000_SDK/Libraries/PeriphDrivers/Source/MAX78000
LIBPATH += ../MAX78000_SDK/Libraries/Boards/MAX78000/$(BOARD)
LIBPATH += ../MAX78000_SDK/Libraries/CMSIS/Device/Maxim/MAX78000/Source

# Libraries
LIBS += PeriphDriver
LIBS += Board
LIBS += arm_cortexM4l_math

# SDK path
MAXIM_PATH := ../MAX78000_SDK

# Include the SDK makefile
include $(MAXIM_PATH)/Libraries/PeriphDrivers/periphdriver.mk

# Additional targets
.PHONY: all clean flash debug info documentation

# Default target
all: $(PROJECT).elf $(PROJECT).hex $(PROJECT).bin

# Flash target
flash: $(PROJECT).bin
	@echo "Flashing $(PROJECT).bin to MAX78000..."
	@$(MAXIM_PATH)/Tools/OpenOCD/openocd -f $(MAXIM_PATH)/Tools/OpenOCD/scripts/interface/cmsis-dap.cfg \
		-f $(MAXIM_PATH)/Tools/OpenOCD/scripts/target/max78000.cfg \
		-c "program $(PROJECT).bin verify reset exit"

# Debug target
debug: $(PROJECT).elf
	@echo "Starting debug session..."
	@$(MAXIM_PATH)/Tools/OpenOCD/openocd -f $(MAXIM_PATH)/Tools/OpenOCD/scripts/interface/cmsis-dap.cfg \
		-f $(MAXIM_PATH)/Tools/OpenOCD/scripts/target/max78000.cfg &
	@arm-none-eabi-gdb $(PROJECT).elf

# Project info
info:
	@echo "MAX78000 EMG Classification System"
	@echo "Target: $(TARGET)"
	@echo "Board: $(BOARD)"
	@echo "Compiler: $(COMPILER)"
	@echo "Project: $(PROJECT)"
	@echo "Sources: $(SRCS)"
	@echo "Include paths: $(IPATH)"

# Generate documentation
documentation:
	@echo "Generating documentation..."
	@doxygen Doxyfile

# Clean target
clean:
	@echo "Cleaning build files..."
	@rm -rf build/
	@rm -f $(PROJECT).elf
	@rm -f $(PROJECT).hex
	@rm -f $(PROJECT).bin
	@rm -f $(PROJECT).map

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build all targets (default)"
	@echo "  clean        - Clean build files"
	@echo "  flash        - Flash firmware to MAX78000"
	@echo "  debug        - Start debug session"
	@echo "  info         - Show project information"
	@echo "  documentation - Generate documentation"
	@echo "  help         - Show this help message"

# Include dependency files
-include $(DEPS)

################################################################################
# CNN Model Integration
################################################################################

# CNN model files (generated by ai8x-synthesis)
CNN_MODEL_DIR := models/generated
CNN_WEIGHTS_FILE := $(CNN_MODEL_DIR)/emg_weights.h
CNN_HEADER_FILE := $(CNN_MODEL_DIR)/emg_cnn_generated.h

# Include CNN model if available
ifneq ($(wildcard $(CNN_WEIGHTS_FILE)),)
PROJ_CFLAGS += -DCNN_MODEL_AVAILABLE
IPATH += $(CNN_MODEL_DIR)
endif

# CNN model generation target
generate-cnn-model:
	@echo "Generating CNN model from ai8x-synthesis..."
	@mkdir -p $(CNN_MODEL_DIR)
	@echo "Please run ai8x-synthesis tools to generate model files"
	@echo "Expected output: $(CNN_WEIGHTS_FILE) and $(CNN_HEADER_FILE)"

################################################################################
# Development and Testing
################################################################################

# Test target
test:
	@echo "Running unit tests..."
	@echo "No unit tests configured yet"

# Lint target
lint:
	@echo "Running code linting..."
	@cppcheck --enable=warning,style,performance,portability \
		--suppress=missingIncludeSystem \
		--inconclusive \
		--std=c99 \
		$(SRCS)

# Format target
format:
	@echo "Formatting code..."
	@astyle --style=allman \
		--indent=spaces=4 \
		--pad-oper \
		--pad-header \
		--unpad-paren \
		--suffix=none \
		--recursive \
		"src/*.c" "src/*.h"

# Size analysis
size: $(PROJECT).elf
	@echo "Memory usage analysis:"
	@arm-none-eabi-size -A $(PROJECT).elf
	@echo ""
	@echo "Section sizes:"
	@arm-none-eabi-size -B $(PROJECT).elf

# Assembly listing
listing: $(PROJECT).elf
	@echo "Generating assembly listing..."
	@arm-none-eabi-objdump -d $(PROJECT).elf > $(PROJECT).lst

################################################################################
# Configuration
################################################################################

# Configuration options
CONFIG_CNN_ACCELERATOR := 1
CONFIG_COMMUNICATION_SPI := 1
CONFIG_DEBUG_UART := 1
CONFIG_LED_STATUS := 1

# Add configuration flags
ifeq ($(CONFIG_CNN_ACCELERATOR), 1)
PROJ_CFLAGS += -DCONFIG_CNN_ACCELERATOR=1
endif

ifeq ($(CONFIG_COMMUNICATION_SPI), 1)
PROJ_CFLAGS += -DCONFIG_COMMUNICATION_SPI=1
endif

ifeq ($(CONFIG_DEBUG_UART), 1)
PROJ_CFLAGS += -DCONFIG_DEBUG_UART=1
endif

ifeq ($(CONFIG_LED_STATUS), 1)
PROJ_CFLAGS += -DCONFIG_LED_STATUS=1
endif

# Print configuration
print-config:
	@echo "Configuration:"
	@echo "  CNN Accelerator: $(CONFIG_CNN_ACCELERATOR)"
	@echo "  Communication SPI: $(CONFIG_COMMUNICATION_SPI)"
	@echo "  Debug UART: $(CONFIG_DEBUG_UART)"
	@echo "  LED Status: $(CONFIG_LED_STATUS)"

################################################################################
# End of Makefile
################################################################################