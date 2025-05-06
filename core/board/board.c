// core/board/board.c

#include "board.h"
#include <stdlib.h>

Board* init_board() {
    Board* board = (Board*) malloc(sizeof(Board));
    if (board == NULL) {
        exit(1);
    }
    board->width = 10;
    board->height = 21;
    for (int x = 0; x < board->width; x++) {
        for (int y = 0; y < board->height + 2; y++) {
            board->state[x][y] = 0;
        }
    }
    return board;
}

