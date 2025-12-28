#ifndef BOARD_H
#define BOARD_H

#include <stdint.h>
#include <stdbool.h>

#define BOARD_WIDTH 10
#define BOARD_HEIGHT 23

typedef struct {
    // rows[0] 是最底部的一行 (y=0)
    // Bit 0 代表最左侧 (x=0)，Bit 9 代表最右侧 (x=9)
    uint16_t rows[BOARD_HEIGHT];
    
    int width;
    int height;
} Board;

Board* init_board();
void free_board(Board* board);
Board* copy_board(Board* board);

static inline int board_get_cell(const Board* board, int x, int y) {
    if (x < 0 || x >= BOARD_WIDTH || y < 0 || y >= BOARD_HEIGHT)
        return 1;
    return (board->rows[y] >> x) & 1;
}

static inline void board_set_cell(Board* board, int x, int y, int val) {
    if (x < 0 || x >= BOARD_WIDTH || y < 0 || y >= BOARD_HEIGHT) return;
    if (val) {
        board->rows[y] |= (1 << x);
    } else {
        board->rows[y] &= ~(1 << x);
    }
}

static inline bool board_is_row_full(const Board* board, int y) {
    return (board->rows[y] & 0x3FF) == 0x3FF;
}

static inline bool board_is_row_empty(const Board* board, int y) {
    return (board->rows[y] & 0x3FF) == 0;
}

#endif