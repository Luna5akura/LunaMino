// core/interact/interact.h

#ifndef INTERACT_H
#define INTERACT_H

#include "../core/game/game.h"

typedef struct {
    float drop_interval;
    int width;
    int height;
    int block_size;
} TetrisConfig;

typedef struct {
    float drop_timer;
    Bool is_game_over;
} TetrisState;

typedef struct {
    TetrisConfig* config;
    TetrisState* state;
    Game* game;
} Tetris;

#endif