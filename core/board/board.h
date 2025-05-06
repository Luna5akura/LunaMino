// core/board/board.h

#ifndef BOARD_H
#define BOARD_H

typedef struct {
    int state[10][23]; // start from Left-down corner (0, 0), x is column, y is row, 0 is empty, 1 is filled
    int width, height;
} Board;

Board* init_board();

#endif