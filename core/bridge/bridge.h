// core/bridge/bridge.h

#ifndef BRIDGE_H
#define BRIDGE_H

#include "../tetris/tetris.h"

// 必须开启强制对齐，否则 int8 后接 int16 会被编译器自动补齐，导致 Python 解包错位
#pragma pack(push, 1)

typedef struct {
    int lines_cleared;
    int damage_sent;
    int attack_type;
    bool is_game_over;
    int b2b_count;
    int combo_count;
} StepResult;

typedef struct {
    int8_t x;
    int8_t y;
    int8_t rotation;
    int8_t landing_height;
    int8_t use_hold;       // 1 byte
    int8_t padding;        // 1 byte (Explicit padding)
    int16_t id;            // 2 bytes
} MacroAction;             // Total: 8 bytes exactly

#define MAX_LEGAL_MOVES 256
typedef struct {
    int count;
    MacroAction moves[MAX_LEGAL_MOVES];
} LegalMoves;

#pragma pack(pop)

// Lifecycle
Tetris* create_tetris(int seed);
void destroy_tetris(Tetris* tetris);
Tetris* clone_tetris(const Tetris* tetris);
void ai_reset_game(Tetris* tetris, int seed);

// Observation
void ai_get_state(const Tetris* tetris, uint8_t* board_buffer, float* ctx_buffer);

// Action
void ai_get_legal_moves(const Tetris* tetris, LegalMoves* out_moves);
StepResult ai_step(Tetris* tetris, int x, int y, int rotation, int use_hold);
void ai_receive_garbage(Tetris* tetris, int lines);

// Visualization
void ai_enable_visualization(Tetris* tetris);
void ai_render(Tetris* tetris);
void ai_close_visualization();

#endif