// core/board/board.c

#include "board.h"
#include <stdlib.h>
#include <string.h>

Board* init_board() {
    Board* board = (Board*)malloc(sizeof(Board));
    if (board == NULL) {
        exit(1);
    }
    memset(board, 0, sizeof(Board));
    
    board->width = BOARD_WIDTH;
    board->height = BOARD_HEIGHT;
    
    return board;
}

void free_board(Board* board) {
    if (board) {
        free(board);
    }
}

Board* copy_board(Board* board) {
    if (!board) return NULL;

    Board* new_board = (Board*)malloc(sizeof(Board));
    if (new_board == NULL) {
        exit(1);
    }
    
    memcpy(new_board, board, sizeof(Board));
    
    return new_board;
}