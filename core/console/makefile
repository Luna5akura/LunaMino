
CC = gcc
CFLAGS = -Wall -g

INCLUDE = -I../board -I../game -I../piece -I../../util

TARGET = tetris

SOURCES = console.c ../board/board.c ../game/game.c ../piece/piece.c ../../util/util.c ../game/bag/bag.c ../game/previews/previews.c

OBJECTS = $(SOURCES:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

clean:
	@echo "Cleaning object files and $(TARGET)..."
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean