// core/tetris/tetris_shared.h
#ifndef TETRIS_SHARED_H
#define TETRIS_SHARED_H

#include "../game/game.h"

typedef struct {
    int fps;
    float gravity;
    float das;
    float arr;
    float soft_drop_gravity;
    float drop_interval;
    float soft_drop_interval;
    float undo_interval;
    float lock_delay;
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
    int pending_attack;
    Bool is_game_over;
} TetrisState;

typedef struct {
    TetrisConfig* config;
    TetrisState* state;
    Game* game;
} Tetris;

// 核心函数声明（无UI依赖）
Tetris* init_tetris(Game* game);
void free_tetris(Tetris* tetris);
Tetris* copy_tetris(Tetris* tetris);
int get_atk(Tetris* tetris);
void receive_garbage_line(Tetris* tetris, int line_count);
void receive_attack(Tetris* tetris, int attack);
void update_clear_rows(Tetris* tetris);
void send_garbage_line(Tetris* tetris, int line_count);

#endif