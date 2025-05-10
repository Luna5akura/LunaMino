// core/tetris/tetris_history/tetris_history.h

#ifndef TETRIS_HISTORY_H
#define TETRIS_HISTORY_H

#include "../tetris.h"

typedef struct {
    int length;
    int current;
    Tetris* tetris_histories[];
} TetrisHistory;

TetrisHistory* init_tetris_history(int max_history_size);
void free_tetris_history(TetrisHistory* history);
void push_history(TetrisHistory* history, Tetris* tetris);
Tetris* pop_history(TetrisHistory* history);


#endif 