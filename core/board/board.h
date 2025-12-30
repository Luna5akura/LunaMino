// core/board/board.h
#ifndef BOARD_H
#define BOARD_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h> // For memset

#define BOARD_WIDTH 10
#define BOARD_HEIGHT 23
#define BOARD_ROW_MASK 0x03FF // 10 bits: 1111111111

typedef struct {
    // 23行 * 2字节 = 46字节
    uint16_t rows[BOARD_HEIGHT];
} Board;

// 初始化
static inline void board_init(Board* board) {
    memset(board->rows, 0, sizeof(board->rows));
}

// 获取单元格 (x, y 使用 int8_t 匹配 Piece 的类型)
static inline uint8_t board_get_cell(const Board* board, int8_t x, int8_t y) {
    // 左右越界：是墙
    if ((uint8_t)x >= BOARD_WIDTH) return 1;
    
    // 底部越界：是墙 (y < 0，经转 uint8_t 后变成很大的数，会包含在下面的判断吗？)
    // 注意：int8_t 的 -1 转 uint8_t 是 255。
    // 如果 y < 0，它是墙。
    if (y < 0) return 1;

    // 顶部越界：通常视为无碰撞（允许方块只是暂存在外部），或者视为墙（不允许）
    // 这取决于你的具体游戏设计。
    if (y >= BOARD_HEIGHT) return 0; // 0 表示空，允许顶部溢出
    // 或者保持原样 return 1; 严防死守

    return (uint8_t)((board->rows[y] >> x) & 1);
}

// 设置单元格 - 无分支优化版
static inline void board_set_cell(Board* board, int8_t x, int8_t y, uint8_t val) {
    // 1. 边界检查 (Branch 1: 必须保留，但已优化为无符号比较)
    if ((uint8_t)x >= BOARD_WIDTH || (uint8_t)y >= BOARD_HEIGHT) return;

    // 2. 准备掩码
    uint16_t mask = 1 << x;

    // 3. 无分支设置位 (Branchless Bit Setting)
    // 逻辑：先用 & ~mask 清除该位，然后用 | (val << x) 设置该位
    // 注意：这里假设 val 只有 0 或 1。如果 val 可能是其他非零值，需改为 (!!val) << x
    board->rows[y] = (board->rows[y] & ~mask) | ((uint16_t)(!!val) << x);
    // board->rows[y] = (board->rows[y] & ~mask) | ((uint16_t)val << x);
}

// 快速判断行满
static inline bool board_is_row_full(const Board* board, int8_t y) {
    // 使用 int8_t 索引
    if ((uint8_t)y >= BOARD_HEIGHT) return false;
    return (board->rows[y] & BOARD_ROW_MASK) == BOARD_ROW_MASK;
}

// 快速判断行空
static inline bool board_is_row_empty(const Board* board, int8_t y) {
    if ((uint8_t)y >= BOARD_HEIGHT) return false;
    return board->rows[y] == 0; 
}

// 声明消行函数
int8_t board_clear_lines(Board* board);

#endif