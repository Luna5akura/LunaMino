// core/board/board.c
#include "board.h"

// 返回消除的行数 (最多一次消4行，int8_t 足够)
int8_t board_clear_lines(Board* board) {
    int8_t clear_count = 0;
    int8_t write_y = 0;
    int8_t read_y = 0;
    
    // 双指针算法：一次遍历完成消除和下落
    for (read_y = 0; read_y < BOARD_HEIGHT; read_y++) {
        // 直接访问 rows 数组通常比调用 board_is_row_full 内联函数更快一点点（减少传参），
        // 但为了代码整洁，调用内联函数完全没问题（编译器会优化）。
        
        // 检查当前行是否已满
        if ((board->rows[read_y] & BOARD_ROW_MASK) == BOARD_ROW_MASK) {
            clear_count++;
            // 这一行被跳过（相当于消除），write_y 不增加
        } else {
            // 如果读指针和写指针位置不同，说明下方有消除，需要搬运数据
            if (write_y != read_y) {
                board->rows[write_y] = board->rows[read_y];
            }
            write_y++;
        }
    }
    
    // 将顶部剩余的行清零
    if (write_y < BOARD_HEIGHT) {
        // 计算需要清零的字节数
        size_t bytes_to_clear = (BOARD_HEIGHT - write_y) * sizeof(uint16_t);
        memset(&board->rows[write_y], 0, bytes_to_clear);
    }
    
    
    return clear_count;
}