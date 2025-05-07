CC = gcc
CFLAGS = -Wall -g

# Raylib 链接选项，根据你的系统可能需要调整
LDFLAGS = -lraylib
# 如果你在 Linux 上，可能还需要添加这些库
LDFLAGS += -lm -lpthread -ldl -lrt

# 如果你使用的是 Wayland，可能需要添加
# LDFLAGS += -lwayland-client -lwayland-cursor -lwayland-egl -lxkbcommon

INCLUDE = -I../board -I../game -I../piece -I../../util

TARGET = tetris

SOURCES = interact.c ../board/board.c ../game/game.c ../piece/piece.c ../../util/util.c ../game/bag/bag.c ../game/previews/previews.c

OBJECTS = $(SOURCES:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

clean:
	@echo "Cleaning object files and $(TARGET)..."
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean