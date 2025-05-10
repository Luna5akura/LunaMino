# Compiler and flags
CC = gcc
CFLAGS = -Wall -g

# Include directories
INCLUDE = -Icore/board -Icore/game -Icore/piece -Iutil

# Common source files
COMMON_SOURCES = core/board/board.c core/game/game.c core/piece/piece.c util/util.c core/game/bag/bag.c core/game/previews/previews.c

# Detect OS
OS := $(shell uname -s 2>/dev/null || echo Windows)

# Add .exe extension for Windows
ifeq ($(OS),Windows)
    EXE_SUFFIX = .exe
    RM = del /Q
    FIXPATH = $(subst /,\,$1)
else
    EXE_SUFFIX =
    RM = rm -f
    FIXPATH = $1
endif

# Console-specific
CONSOLE_TARGET = tetris-console$(EXE_SUFFIX)
CONSOLE_SOURCES = core/console/console.c $(COMMON_SOURCES)
CONSOLE_OBJECTS = $(CONSOLE_SOURCES:.c=.o)

# Raylib-specific
RAYLIB_TARGET = tetris-raylib$(EXE_SUFFIX)
RAYLIB_SOURCES = core/tetris/tetris.c core/tetris/tetris_ui/tetris_ui.c core/tetris/tetris_history/tetris_history.c $(COMMON_SOURCES)
RAYLIB_OBJECTS = $(RAYLIB_SOURCES:.c=.o)

# Platform-specific LDFLAGS for raylib
ifeq ($(OS),Windows)
    RAYLIB_LDFLAGS = -lraylib -lopengl32 -lgdi32 -lwinmm
else
    RAYLIB_LDFLAGS = -lraylib -lm -lpthread -ldl -lrt
endif

# Default target: build both
all: $(CONSOLE_TARGET) $(RAYLIB_TARGET)

# Console target
$(CONSOLE_TARGET): $(CONSOLE_OBJECTS)
	$(CC) $(CONSOLE_OBJECTS) -o $(call FIXPATH,$(CONSOLE_TARGET))

# Raylib target
$(RAYLIB_TARGET): $(RAYLIB_OBJECTS)
	$(CC) $(RAYLIB_OBJECTS) -o $(call FIXPATH,$(RAYLIB_TARGET)) $(RAYLIB_LDFLAGS)

# Object file rules
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $(call FIXPATH,$@)

# Clean up
clean:
	@echo "Cleaning object files and targets..."
	$(RM) $(call FIXPATH,$(CONSOLE_OBJECTS)) $(call FIXPATH,$(RAYLIB_OBJECTS)) $(call FIXPATH,$(CONSOLE_TARGET)) $(call FIXPATH,$(RAYLIB_TARGET))

# Phony targets
.PHONY: all clean console raylib

# Convenience targets for building individually
console: $(CONSOLE_TARGET)
raylib: $(RAYLIB_TARGET)