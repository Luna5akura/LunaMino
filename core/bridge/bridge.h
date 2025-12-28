// core/bridge/bridge.h
#ifndef BRIDGE_H
#define BRIDGE_H

#include "../tetris/tetris.h"

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
    int rotation; // 0-3
    int landing_height;
    int use_hold;
} MacroAction;
#define MAX_LEGAL_MOVES 200
typedef struct {
    int count;
    MacroAction moves[MAX_LEGAL_MOVES];
} LegalMoves;
// AI / Python Export Functions

Tetris* create_tetris(int seed);
void destroy_tetris(Tetris* tetris);
Tetris* clone_tetris(const Tetris* tetris);


void ai_reset_game(Tetris* tetris, int seed);
void ai_get_state(const Tetris* tetris, int* board_buffer, int* queue_buffer, int* hold_buffer, int* meta_buffer);
void ai_get_legal_moves(const Tetris* tetris, LegalMoves* out_moves);
StepResult ai_step(Tetris* tetris, int x, int rotation, int use_hold);
void ai_receive_garbage(Tetris* tetris, int lines);
// 【新增】可视化接口
void ai_enable_visualization(Tetris* tetris);
void ai_render(Tetris* tetris);
void ai_close_visualization();
#endif