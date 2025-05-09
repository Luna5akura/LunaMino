// core/board/board.h

#ifndef BOARD_H
#define BOARD_H

#define BOARD_WIDTH 10
#define BOARD_HEIGHT 23

typedef struct {
    int state[BOARD_WIDTH][BOARD_HEIGHT]; // start from Left-down corner (0, 0), x is column, y is row, 0 is empty, 1 is filled
    int width, height;
} Board;

Board* init_board();

#endif