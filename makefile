# ==========================================
# Tetris Project Makefile (Updated)
# ==========================================
CC = gcc
# 编译选项: 添加 -g for debug if needed
CFLAGS = -Wall -O3 -fPIC -march=native
# 头文件路径 (添加 core/battle if needed)
INCLUDE = -Icore/board -Icore/game -Icore/piece -Iutil -Iai -Icore/tetris -Icore/battle -Icore/console

# 核心源文件 (包含所有逻辑 + 新接口 in game.c)
LOGIC_SOURCES = core/board/board.c \
                core/game/game.c \
                core/game/bag/bag.c \
                core/game/previews/previews.c \
                core/piece/piece.c \
                util/util.c \
                core/tetris/tetris.c \
                core/tetris/tetris_ui/tetris_ui.c

COMMON_SOURCES = core/board/board.c \
core/game/game.c \
core/game/bag/bag.c \
core/game/previews/previews.c \
core/piece/piece.c \
util/util.c \
core/tetris/tetris.c \
core/tetris/tetris_ui/tetris_ui.c \
core/tetris/tetris_history/tetris_history.c \
core/battle/battle.c \
core/console/console.c

# Bridge for AI: 如果添加 ai/bridge.c, 包含在这里; 否则 game.c 已足够

# 目标文件名
SHARED_LIB = libtetris.so
RAYLIB_EXE = tetris-raylib
CONSOLE_EXE = tetris-console
TRAIN_EXE = tetris-train  # 如果有C训练入口，可添加
# 链接选项
SHARED_LDFLAGS = -shared -lm -lraylib
# Raylib 链接 (调整为您的系统)
RAYLIB_LDFLAGS = -lraylib -lm -lpthread -ldl -lrt
# ==========================================
# Build Targets
# ==========================================
all: shared raylib console

BRIDGE_SOURCES = $(LOGIC_SOURCES)  # Override for shared
BRIDGE_OBJECTS = $(BRIDGE_SOURCES:.c=.o)

# 编译共享库的目标文件规则
shared: SHARED_CFLAGS = $(CFLAGS) -DSHARED_LIB=1
shared: $(BRIDGE_OBJECTS)
	@echo "Building Shared Library for Python..."
	$(CC) $(SHARED_CFLAGS) $(BRIDGE_OBJECTS) -o $(SHARED_LIB) $(SHARED_LDFLAGS)
	@echo "Done: $(SHARED_LIB)"

core/tetris/tetris.o: core/tetris/tetris.c
	$(CC) $(CFLAGS) -DSHARED_LIB=1 -c $< -o $@

# 2. 编译 Raylib 游戏版本
raylib:
	@echo "Building Raylib Game..."
	$(CC) $(CFLAGS) $(INCLUDE) $(COMMON_SOURCES) -o $(RAYLIB_EXE) $(RAYLIB_LDFLAGS)
	@echo "Done: $(RAYLIB_EXE)"

# 3. 编译控制台测试版
console:
	@echo "Building Console Test..."
	$(CC) $(CFLAGS) $(INCLUDE) $(COMMON_SOURCES) -o $(CONSOLE_EXE) -lm
	@echo "Done: $(CONSOLE_EXE)"

# 编译通用的 .o 文件规则
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

# 清理
clean:
	rm -f $(BRIDGE_OBJECTS) $(SHARED_LIB) $(RAYLIB_EXE) $(CONSOLE_EXE)
	rm -f core/board/*.o core/game/*.o core/piece/*.o util/*.o ai/*.o core/tetris/*.o core/battle/*.o core/console/*.o

.PHONY: all shared raylib console clean