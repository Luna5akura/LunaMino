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
    TetrisConfig* config;
    TetrisState* state;
    Game* game;
} Tetris;

// ==========================================
// AI Interface Definitions (新增部分 - 必须放在这里)
// ==========================================

// 用于返回给 Python 的单步执行结果
typedef struct {
    int lines_cleared;
    int damage_sent;     
    int attack_type;     
    Bool is_game_over;
    int b2b_count;
    int combo_count;     
} StepResult;

// 用于描述一个合法的落点（宏动作）
typedef struct {
    int x;
    int y;
    int rotation;        // 0-3
    int landing_height;  
    int use_hold;
} MacroAction;

#define MAX_LEGAL_MOVES 200

typedef struct {
    int count;
    MacroAction moves[MAX_LEGAL_MOVES];
} LegalMoves;

// ==========================================
// Functions
// ==========================================

Tetris* init_tetris(Game* game);
void free_tetris(Tetris* tetris);
Tetris* copy_tetris(Tetris* tetris);

// UI 依赖函数
#ifndef SHARED_LIB
void update_drop_timer(Tetris* tetris);
void detect_left_or_right(Tetris* tetris, InputControl controls);
Tetris* detect_input(Tetris* tetris, void* tetris_history, InputControl controls);
void flush_lock_timer(Tetris* tetris);
void detect_hard_drop(Tetris* tetris);
#endif

// 核心攻击函数
int get_atk(Tetris* tetris);
void receive_garbage_line(Tetris* tetris, int line_count);
void receive_attack(Tetris* tetris, int attack);
void update_clear_rows(Tetris* tetris);
void send_garbage_line(Tetris* tetris, int line_count);

// ==========================================
// AI / Python Export Functions (新增声明)
// ==========================================

void ai_reset_game(Tetris* tetris, int seed);
void ai_get_state(Tetris* tetris, int* board_buffer, int* queue_buffer, int* hold_buffer, int* meta_buffer);
void ai_get_legal_moves(Tetris* tetris, LegalMoves* out_moves);
StepResult ai_step(Tetris* tetris, int x, int rotation, int use_hold);
void ai_receive_garbage(Tetris* tetris, int lines);

void ai_reset_game(Tetris* tetris, int seed);
void ai_get_state(Tetris* tetris, int* board_buffer, int* queue_buffer, int* hold_buffer, int* meta_buffer);
void ai_get_legal_moves(Tetris* tetris, LegalMoves* out_moves);
StepResult ai_step(Tetris* tetris, int x, int rotation, int use_hold);

void ai_receive_garbage(Tetris* tetris, int lines);

// 【新增】可视化接口
void ai_enable_visualization(Tetris* tetris);
void ai_render(Tetris* tetris);
void ai_close_visualization();

#endif