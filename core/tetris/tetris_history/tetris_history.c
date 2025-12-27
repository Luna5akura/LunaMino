// core/tetris/tetris_history/tetris_history.c

#include "tetris_history.h"
#include <stdlib.h>
#include <string.h> // 需要包含 string.h 以使用 memset

TetrisHistory* init_tetris_history(int max_history_size) {
    TetrisHistory* history = (TetrisHistory*)malloc(sizeof(TetrisHistory) + sizeof(Tetris*) * max_history_size);
    history->length = max_history_size;
    history->current = 0;
    // 【必须有这一行】
    memset(history->tetris_histories, 0, sizeof(Tetris*) * max_history_size); 
    return history;
}

void free_tetris_history(TetrisHistory* history) {
    if (history == NULL) return;
    for (int i = 0; i < history->length; i++) {
        // 由于我们上面 memset 了，未使用的位置是 NULL
        // 修改后的 free_tetris 会安全处理 NULL
        free_tetris(history->tetris_histories[i]);
    }
    free(history);
}

void push_history(TetrisHistory* history, Tetris* tetris) {
    history->current = (history->current + 1) % history->length;
    
    // 如果该位置已有旧的历史记录，先释放它，防止内存泄漏
    if (history->tetris_histories[history->current] != NULL) {
        free_tetris(history->tetris_histories[history->current]);
    }
    
    history->tetris_histories[history->current] = copy_tetris(tetris);
}

Tetris* pop_history(TetrisHistory* history) {
    if (history->current == 0) {
        return NULL;
    }
    
    // 检查当前位置是否有内容（虽然逻辑上应该有，但为了安全）
    if (history->tetris_histories[history->current] == NULL) {
        return NULL; 
    }

    Tetris* tetris = history->tetris_histories[history->current];
    history->tetris_histories[history->current] = NULL; // 弹出后置空
    
    // 环形缓冲区回退逻辑
    history->current = (history->current - 1 + history->length) % history->length;

    return tetris;
}