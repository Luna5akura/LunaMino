# ==========================================
# Tetris AI Bridge Makefile (Shared Only)
# ==========================================

CC = gcc

# 1. 编译选项
# -fPIC: 生成位置无关代码 (必须用于动态库)
# -DSHARED_LIB=1: 告知代码当前是编译动态库模式
# -g: 保留调试符号 (Python调用出错时能看到栈信息)
CFLAGS = -Wall -O3 -fPIC -g -DSHARED_LIB=1

# 2. 头文件路径
# 注意：添加了 tetris_ui 的路径，确保 bridge.c 能找到 ui 头文件
INCLUDE = -Icore/board \
          -Icore/game \
          -Icore/piece \
          -Iutil \
          -Icore/tetris \
          -Icore/tetris/tetris_ui \
          -Icore/bridge

# 3. 链接选项
# -lraylib: 必须链接 raylib，因为 bridge.c 和 tetris_ui.c 用到了它
# -lm: 数学库
LDFLAGS = -shared -lm -lraylib

# 4. 源文件列表
# 包含了完整的游戏逻辑 + UI + Bridge
SOURCES = core/board/board.c \
          core/game/game.c \
          core/game/bag/bag.c \
          core/game/previews/previews.c \
          core/piece/piece.c \
          core/tetris/tetris.c \
          core/tetris/tetris_ui/tetris_ui.c \
          core/bridge/bridge.c

# 5. 目标文件 (.c -> .o)
OBJECTS = $(SOURCES:.c=.o)

# 6. 输出文件名
TARGET = libtetris.so

# ==========================================
# Build Rules
# ==========================================

all: $(TARGET)

# 链接规则
$(TARGET): $(OBJECTS)
	@echo "Linking $(TARGET)..."
	$(CC) $(CFLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)
	@echo "Build Complete: $(TARGET)"

# 编译规则 (通用规则，适用于所有 .c 文件)
# 确保在这里加上 $(INCLUDE)
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

# 清理规则
clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean