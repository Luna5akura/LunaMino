// core/tetris/tetris.h
#ifndef TETRIS_H
#define TETRIS_H
#include "../game/game.h"
#ifndef SHARED_LIB
// 新增：按键控制结构体 - 只在非共享库模式
typedef struct {
    int left, right, up, down;
    int hard_drop;
    int rotate_cw, rotate_ccw, rotate_180;
    int hold, undo, restart;
} InputControl;
#endif
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
    int is_game_over;
} TetrisState;
typedef struct {
    TetrisConfig config;
    TetrisState state;
    Game game;
} Tetris;
// Functions
Tetris* tetris_init(const GameConfig* game_config);
void tetris_free(Tetris* tetris);
Tetris* tetris_copy(const Tetris* tetris);
// UI 依赖函数
#ifndef SHARED_LIB
void tetris_update_drop_timer(Tetris* tetris);
void tetris_detect_left_or_right(Tetris* tetris, InputControl controls);
Tetris* tetris_detect_input(Tetris* tetris, void* tetris_history, InputControl controls);
void tetris_flush_lock_timer(Tetris* tetris);
void tetris_detect_hard_drop(Tetris* tetris);
#endif
// 核心攻击函数
int tetris_get_atk(Tetris* tetris);
void tetris_receive_garbage_line(Tetris* tetris, int line_count);
void tetris_receive_attack(Tetris* tetris, int attack);
void tetris_update_clear_rows(Tetris* tetris);
void tetris_send_garbage_line(Tetris* tetris, int line_count);
#endif