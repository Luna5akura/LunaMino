// core/tetris/tetris_history/tetris_history.c

#include "tetris_history.h"
#include <stdlib.h>

TetrisHistory* init_tetris_history(int max_history_size) {
    TetrisHistory* history = (TetrisHistory*)malloc(sizeof(TetrisHistory) + sizeof(Tetris*) * max_history_size);

    history->length = max_history_size;
    history->current = 0;

    return history;
}

void free_tetris_history(TetrisHistory* history) {
    for (int i = 0; i < history->length; i++) {
        free_tetris(history->tetris_histories[i]);
    }
    free(history);
}

void push_history(TetrisHistory* history, Tetris* tetris) {
    history->current = (history->current + 1) % history->length;
    history->tetris_histories[history->current] = copy_tetris(tetris);
}

Tetris* pop_history(TetrisHistory* history) {
    if (history->current == 0) {
        return NULL;
    }

    Tetris* tetris = history->tetris_histories[history->current];
    if (tetris == NULL) return NULL;
    history->tetris_histories[history->current] = NULL;
    history->current = (history->current - 1 + history->length) % history->length;

    return tetris;
}
