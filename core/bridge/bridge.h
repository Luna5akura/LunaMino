// core/bridge/bridge.h

#ifndef BRIDGE_H
#define BRIDGE_H

#include "../tetris/tetris.h"

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
    bool use_hold;
} MacroAction;

#define MAX_LEGAL_MOVES 256

typedef struct {
    int count;
    MacroAction moves[MAX_LEGAL_MOVES];
} LegalMoves;

// Lifecycle
Tetris* create_tetris(int seed);
void destroy_tetris(Tetris* tetris);
Tetris* clone_tetris(const Tetris* tetris);
void ai_reset_game(Tetris* tetris, int seed);

// Observation
// board_buffer: 200 ints (10x20)
// queue_buffer: 5 ints
// hold_buffer: 1 int (-1 if empty)
// meta_buffer: [b2b, combo, can_hold, current_piece_type, pending_garbage]
void ai_get_state(const Tetris* tetris, int* board_buffer, int* queue_buffer, int* hold_buffer, int* meta_buffer);

// Action
// Returns detailed result of the step
void ai_get_legal_moves(const Tetris* tetris, LegalMoves* out_moves);
StepResult ai_step(Tetris* tetris, int x, int y, int rotation, int use_hold);
void ai_receive_garbage(Tetris* tetris, int lines);

// Visualization
void ai_enable_visualization(Tetris* tetris);
void ai_render(Tetris* tetris);
void ai_close_visualization();

#endif