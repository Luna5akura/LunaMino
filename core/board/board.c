// core/board/board.c

#include "board.h"
#include <stdlib.h>

#define BOARD_WIDTH 10
#define BOARD_HEIGHT 23

Board* init_board() {
    Board* board = (Board*) malloc(sizeof(Board));
    if (board == NULL) {
        exit(1);
    }
    board->width = BOARD_WIDTH;
    board->height = BOARD_HEIGHT;
    for (int x = 0; x < board->width; x++) {
        for (int y = 0; y < board->height; y++) {
            board->state[x][y] = 0;
        }
    }
    return board;
}

void free_board(Board* board) {
    free(board);
}

Board* copy_board(Board* board) {
    Board* new_board = (Board*) malloc(sizeof(Board));
    if (new_board == NULL) {
        exit(1);
    }
    memcpy(new_board, board, sizeof(Board));
    return new_board;
}
