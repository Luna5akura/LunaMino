// core/tetris/tetris.h
#ifndef TETRIS_H
#define TETRIS_H

#include <stdbool.h>
#include "../game/game.h"

typedef struct {
    int fps;
    float gravity;
    float das;
    float arr;
    float soft_drop_gravity;
    
    // Calculated intervals
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
    
    // Input states
    bool is_left_pressed;
    bool is_right_pressed;
    bool is_soft_drop_pressed;
    
    // Game states
    bool is_grounded;
    bool is_update_clear_rows_needed;
    
    // Stats & Combat
    AttackType attack_type;
    bool is_pc;
    int b2b_count;       // -1 表示无 B2B, >=0 表示 B2B 连击数
    int atk_count;       // 总发出攻击
    int pending_attack;  // 缓冲槽中的垃圾行
    
    bool is_game_over;
} TetrisState;

typedef struct {
    TetrisConfig config;
    TetrisState state;
    Game game;
} Tetris;

// Lifecycle
Tetris* tetris_init(const GameConfig* game_config);
void tetris_free(Tetris* tetris);
Tetris* tetris_copy(const Tetris* tetris);

// Logic
int tetris_get_atk(Tetris* tetris);
void tetris_receive_garbage_line(Tetris* tetris, int line_count);
void tetris_receive_attack(Tetris* tetris, int attack);
void tetris_update_clear_rows(Tetris* tetris);
void tetris_send_garbage_line(Tetris* tetris, int line_count);

#endif