// core/tetris/tetris.h

#ifndef TETRIS_H
#define TETRIS_H

#include "../core/game/game.h"

typedef struct {

    int fps;
    float gravity; // block per frame
    float das; // ms
    float arr; // ms
    float soft_drop_gravity; // block per frame
    float drop_interval;
    float soft_drop_interval;
    float undo_interval;
    float lock_delay; // ms
    int reset_lock_times_limit;
} TetrisConfig;

typedef struct {
    float drop_timer;
    float das_timer;
    float arr_timer;
    float soft_drop_timer;
    float undo_timer;
    float lock_timer;
    int lock_times_left;
    Bool is_left_pressed;
    Bool is_right_pressed;
    Bool is_soft_drop_pressed;
    Bool is_grounded;
    Bool is_update_clear_rows_needed;
    AttackType attack_type;
    Bool is_pc;
    int b2b_count;
    int atk_count;
    Bool is_game_over;
} TetrisState;

typedef struct {
    TetrisConfig* config;
    TetrisState* state;
    Game* game;
} Tetris;

void free_tetris(Tetris* tetris);
Tetris* copy_tetris(Tetris* tetris);

#endif