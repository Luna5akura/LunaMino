// core/tetris/tetris.c
#include "tetris.h"
#ifndef SHARED_LIB
#include <raylib.h>
#include "tetris_ui/tetris_ui.h"
#include "tetris_history/tetris_history.h"
#else
#include "../game/game.h"
#endif
#include <string.h>
#include <stdlib.h>
#define DEFAULT_FPS 60
#define DEFAULT_GRAVITY 1.0f / 60.0f
#define DEFAULT_DAS 100.0f
#define DEFAULT_ARR 0.0f
#define DEFAULT_SOFT_DROP_GRAVITY 0.0f
#define DEFAULT_UNDO_INTERVAL 150.0f
#define DEFAULT_LOCK_DELAY 500.0f
#define DEFAULT_RESET_LOCK_TIMES_LIMIT 15

// 攻击表定义
const int ATK_TABLE[21][9] = {
    {0, 1, 2, 4, 0, 2, 1, 4, 6},
    {0, 1, 2, 5, 0, 2, 1, 5, 7},
    {1, 1, 3, 6, 1, 3, 1, 6, 9},
    {1, 1, 3, 7, 1, 3, 1, 7, 10},
    {1, 2, 4, 8, 1, 4, 2, 8, 12},
    {1, 2, 4, 9, 1, 4, 2, 9, 13},
    {2, 2, 5, 10, 2, 5, 2, 10, 15},
    {2, 2, 5, 11, 2, 5, 2, 11, 16},
    // {2, 3, 6, 12, 2, 6, 3, 12, 18},
    {2, 3, 6, 13, 2, 6, 3, 13, 19},
    {2, 3, 7, 14, 2, 7, 3, 14, 21},
    {2, 3, 7, 15, 2, 7, 3, 15, 22},
    {2, 4, 8, 16, 2, 8, 4, 16, 24},
    {2, 4, 8, 17, 2, 8, 4, 17, 25},
    {2, 4, 9, 18, 2, 9, 4, 18, 27},
    {2, 4, 9, 19, 2, 9, 4, 19, 28},
    {3, 5, 10, 20, 3, 10, 5, 20, 30},
    {3, 5, 10, 21, 3, 10, 5, 21, 31},
    {3, 5, 11, 22, 3, 11, 5, 22, 33},
    {3, 5, 11, 23, 3, 11, 5, 23, 34},
    {3, 6, 12, 24, 3, 12, 6, 24, 36}
};
const int ATK_TABLE_B2B1[21][6] = {
    {5, 1, 3, 2, 5, 7},
    {6, 1, 3, 2, 6, 8},
    {7, 1, 4, 3, 7, 10},
    {8, 1, 5, 3, 8, 12},
    {10, 2, 6, 4, 10, 14},
    {11, 2, 6, 4, 11, 15},
    {12, 2, 7, 5, 12, 17},
    {13, 2, 8, 5, 13, 19},
    {15, 3, 9, 6, 15, 21},
    {16, 3, 9, 6, 16, 22},
    {17, 3, 10, 7, 17, 24},
    {18, 3, 11, 7, 18, 26},
    {20, 4, 12, 8, 20, 28},
    {21, 4, 12, 8, 21, 29},
    {22, 4, 13, 9, 22, 31},
    {23, 4, 14, 9, 23, 33},
    {25, 5, 15, 10, 25, 35},
    {26, 5, 15, 10, 26, 36},
    {27, 5, 16, 11, 27, 38},
    {28, 5, 17, 11, 28, 40},
    {30, 6, 18, 12, 30, 42},
};

Tetris* tetris_init(const GameConfig* game_config) {
    Tetris* tetris = (Tetris*)malloc(sizeof(Tetris));
    if (tetris == NULL) return NULL;
    memset(tetris, 0, sizeof(Tetris));

    // Config
    tetris->config.fps = DEFAULT_FPS;
    tetris->config.gravity = DEFAULT_GRAVITY;
    tetris->config.das = DEFAULT_DAS;
    tetris->config.arr = DEFAULT_ARR;
    tetris->config.soft_drop_gravity = DEFAULT_SOFT_DROP_GRAVITY;
    tetris->config.drop_interval = 1.0f / (DEFAULT_GRAVITY * DEFAULT_FPS);
    tetris->config.soft_drop_interval = DEFAULT_SOFT_DROP_GRAVITY;
    tetris->config.undo_interval = DEFAULT_UNDO_INTERVAL;
    tetris->config.lock_delay = DEFAULT_LOCK_DELAY;
    tetris->config.reset_lock_times_limit = DEFAULT_RESET_LOCK_TIMES_LIMIT;

    // State
    tetris->state.drop_timer = 0.0f;
    tetris->state.das_timer = 0.0f;
    tetris->state.arr_timer = 0.0f;
    tetris->state.soft_drop_timer = 0.0f;
    tetris->state.undo_timer = 0.0f;
    tetris->state.lock_timer = 0.0f;
    tetris->state.lock_times_left = tetris->config.reset_lock_times_limit;
    tetris->state.is_left_pressed = false;
    tetris->state.is_right_pressed = false;
    tetris->state.is_soft_drop_pressed = false;
    tetris->state.is_grounded = false;
    tetris->state.is_update_clear_rows_needed = false;
    tetris->state.attack_type = ATK_NONE;
    tetris->state.is_pc = false;
    tetris->state.b2b_count = -1;
    tetris->state.atk_count = 0;
    tetris->state.pending_attack = 0;
    tetris->state.is_game_over = false;

    // Game
    tetris->game = *game_init(game_config);

    return tetris;
}

void tetris_free(Tetris* tetris) {
    if (tetris) {
        free(tetris);
    }
}

Tetris* tetris_copy(const Tetris* src) {
    if (!src) return NULL;
    Tetris* dest = (Tetris*)malloc(sizeof(Tetris));
    if (dest == NULL) return NULL;
    *dest = *src;
    dest->game = *game_copy(&src->game);
    return dest;
}

static void reset_lock_times_left(Tetris* tetris) {
    tetris->state.lock_times_left = tetris->config.reset_lock_times_limit;
    tetris->state.lock_timer = 0.0f;
}

void tetris_flush_lock_timer(Tetris* tetris) {
    tetris->state.lock_timer = 0.0f;
    tetris->state.lock_times_left -= 1;
    if (tetris->state.lock_times_left < 0) {
        reset_lock_times_left(tetris);
        tetris->state.is_update_clear_rows_needed = true;
        tetris->state.is_grounded = false;
    }
}


// UI / Control Logic (Non-Shared Only)
#ifndef SHARED_LIB
void tetris_update_drop_timer(Tetris* tetris) {
    if (tetris->state.is_grounded) {
        tetris->state.lock_timer += GetFrameTime();
        if (tetris->state.lock_timer * 1000 >= tetris->config.lock_delay) {
            reset_lock_times_left(tetris);
            tetris->state.is_update_clear_rows_needed = true;
            tetris->state.is_grounded = false;
        }
    } else {
        tetris->state.drop_timer += GetFrameTime();
        if (tetris->state.drop_timer < tetris->config.drop_interval) return;
        tetris->state.drop_timer = 0.0f;
        game_try_move(&tetris->game, MOVE_DOWN);
        tetris->state.is_grounded = is_grounded(&tetris->game); // assume function in game
    }
}

void detect_left_or_right(Tetris* tetris, InputControl controls) {
    Game game = tetris->game;
    if (IsKeyPressed(controls.left) || IsKeyPressed(controls.right)) {
        tetris->state.das_timer = 0;
        Bool is_successful = FALSE;
        if (IsKeyPressed(controls.left)) is_successful = try_move_piece(game, MOVE_LEFT);
        if (IsKeyPressed(controls.right)) is_successful = try_move_piece(game, MOVE_RIGHT);
        if (!is_successful) return;
        tetris->state.is_grounded = is_grounded(game);
        if (tetris->state.is_grounded) flush_lock_timer(tetris);
    }
    else {
        tetris->state.das_timer += GetFrameTime() * 1000;
        if (tetris->state.das_timer < tetris->config.das) return;
        tetris->state.arr_timer += GetFrameTime() * 1000;
        Bool is_looping = TRUE;
        int move_count = -1;
        while (tetris->state.arr_timer >= tetris->config.arr && is_looping) {
            if (IsKeyDown(controls.left)) is_looping = try_move_piece(game, MOVE_LEFT);
            if (IsKeyDown(controls.right)) is_looping = try_move_piece(game, MOVE_RIGHT);
            move_count++;
            tetris->state.arr_timer -= tetris->config.arr;
        }
        tetris->state.is_grounded = is_grounded(&tetris->game);
        if (tetris->state.is_grounded && move_count > 0) flush_lock_timer(tetris);
    }
}

void detect_soft_drop(Tetris* tetris, int key_soft_drop) {
    Game game = tetris->game;
    if (IsKeyDown(key_soft_drop)) {
        if (tetris->config.soft_drop_gravity <= 0.0f) {
            while (try_move_piece(game, MOVE_SOFT_DROP));
            tetris->state.soft_drop_timer = 0.0f;
            return;
        }
        if (IsKeyPressed(key_soft_drop)) {
            try_move_piece(game, MOVE_SOFT_DROP);
            tetris->state.soft_drop_timer = 0.0f;
        }
        else {
            tetris->state.soft_drop_timer += GetFrameTime() * 1000;
            Bool is_looping = TRUE;
            float interval = tetris->config.soft_drop_interval;
            if (interval <= 0.0001f) interval = 1.0f;
            while (tetris->state.soft_drop_timer >= tetris->config.soft_drop_gravity && is_looping) {
                is_looping = try_move_piece(game, MOVE_SOFT_DROP);
                tetris->state.soft_drop_timer -= interval;
            }
        }
    }
}

void detect_rotate(Tetris* tetris, RotationAction action) {
    Game game = tetris->game;
    Bool is_successful = try_rotate_piece(game, action);
    if (!is_successful) return;
    tetris->state.is_grounded = is_grounded(game);
    if (tetris->state.is_grounded) flush_lock_timer(tetris);
}

void detect_hold(Tetris* tetris) {
    Game game = tetris->game;
    tetris->state.is_game_over = try_hold_piece(game);
    tetris->state.drop_timer = 0.0f;
    tetris->state.is_grounded = FALSE;
    reset_lock_times_left(tetris);
}

void detect_hard_drop(Tetris* tetris) {
    Game game = tetris->game;
    try_move_piece(game, MOVE_HARD_DROP);
    tetris->state.drop_timer = 0.0f;
    tetris->state.is_grounded = FALSE;
    reset_lock_times_left(tetris);
    tetris->state.is_update_clear_rows_needed = TRUE;
}

Tetris* detect_undo(Tetris* tetris, TetrisHistory* tetris_history) {
    if (tetris->state.undo_timer * 1000 < tetris->config.undo_interval) return tetris;
    tetris->state.undo_timer = 0.0f;
    Tetris* pop_tetris = pop_history(tetris_history);
    if (pop_tetris == NULL) return tetris;
    pop_tetris->state.undo_timer = 0.0f;
    free_tetris(tetris);
    return pop_tetris;
}

Tetris* detect_input(Tetris* tetris, void* tetris_history_ptr, InputControl controls) {
    TetrisHistory* tetris_history = (TetrisHistory*)tetris_history_ptr;
    if (IsKeyDown(controls.left) || IsKeyDown(controls.right)) detect_left_or_right(tetris, controls);
    else tetris->state.das_timer = 0.0f;
   
    if (IsKeyDown(controls.down)) detect_soft_drop(tetris, controls.down);
   
    if (IsKeyPressed(controls.rotate_ccw)) detect_rotate(tetris, ROTATE_CCW);
    if (IsKeyPressed(controls.rotate_cw)) detect_rotate(tetris, ROTATE_CW);
    if (IsKeyPressed(controls.rotate_180)) detect_rotate(tetris, ROTATE_180);
   
    if (IsKeyPressed(controls.hold)) detect_hold(tetris);
    if (IsKeyPressed(controls.hard_drop)) detect_hard_drop(tetris);
   
    if (controls.restart != 0 && IsKeyDown(controls.restart)) tetris = restart_game(tetris);
    if (IsKeyDown(controls.undo)) tetris = detect_undo(tetris, tetris_history);
   
    return tetris;
}
#endif // SHARED_LIB


// ============================================================================
// Core Logic (Attack, Clear Rows)
// ============================================================================

int get_s2_atk(Tetris* tetris) {
    int total_atk = 0;
    AttackType type = tetris->state.attack_type;
    int ren = tetris->game.state.ren;
    if (ren < 0) ren = 0;
    if (ren > 20) ren = 20;

    Bool keeps_b2b = FALSE;
    if (type == ATK_TETRIS ||
        (type >= ATK_TSMS && type <= ATK_TST) ||
        (type >= ATK_ISS)) { 
        keeps_b2b = TRUE;
    }
    
    int base_dmg = 0;
   
    if (type >= ATK_SINGLE && type <= ATK_TST) {
        if (tetris->state.b2b_count > 0 && keeps_b2b) {
             if (type >= ATK_TETRIS) {
                 int idx = (int)type - (int)ATK_TETRIS;
                 if (idx >= 0 && idx < 6)
                     base_dmg = ATK_TABLE_B2B1[ren][idx];
             }
        } else {
             int idx = (int)type - (int)ATK_SINGLE;
             if (idx >= 0 && idx < 9)
                 base_dmg = ATK_TABLE[ren][idx];
        }
    }
    else if (type >= ATK_ISS) {
        base_dmg = 0;
        if (tetris->state.b2b_count > 0) base_dmg += 1;
    }

    if (!keeps_b2b && tetris->state.b2b_count > 4) {
        total_atk += tetris->state.b2b_count;
    }

    if (tetris->state.is_pc) {
        total_atk += 10;
    }
    total_atk += base_dmg;
    return total_atk;
}

int tetris_get_atk(Tetris* tetris) {
    if (tetris->state.attack_type == ATK_NONE) return 0;
    return get_s2_atk(tetris);
}

void tetris_receive_garbage_line(Tetris* tetris, int line_count) {
    // update to new board_set_cell etc.
    Board* board = &tetris->game.board;
    int hole_column = magic_random() % board->width;
    // shift rows up
    for (int y = 0; y < board->height - line_count; y++) {
        board->rows[y + line_count] = board->rows[y];
    }
    // add garbage
    for (int y = 0; y < line_count; y++) {
        uint16_t row = 0x03FF; // full
        row &= ~(1 << hole_column); // hole
        board->rows[y] = row;
    }
}

void tetris_receive_attack(Tetris* tetris, int attack) {
    tetris->state.pending_attack += attack;
}

void tetris_update_clear_rows(Tetris* tetris) {
    // similar to provided, but use new functions
    tetris->state.attack_type = game_get_attack_type(&tetris->game);
    tetris->state.is_pc = game_is_perfect_clear(&tetris->game);
    int attack = tetris_get_atk(tetris);
    tetris->state.atk_count = attack;
    game_next_step(&tetris->game);
    int clear_count = clear_rows(&tetris->game.board);
    if (clear_count == 0) tetris->game.state.ren = -1;
    else tetris->game.state.ren++;
    // b2b update
    if (tetris->state.attack_type == ATK_SINGLE || ATK_DOUBLE || ATK_TRIPLE) tetris->state.b2b_count = -1;
    else if (tetris->state.attack_type != ATK_NONE) tetris->state.b2b_count++;
    // garbage logic
    if (clear_count > 0) {
        if (tetris->state.pending_attack) {
            if (attack > tetris->state.pending_attack) {
                tetris_send_garbage_line(tetris, attack - tetris->state.pending_attack);
                tetris->state.pending_attack = 0;
            } else {
                tetris->state.pending_attack -= attack;
            }
        } else {
            tetris_send_garbage_line(tetris, attack);
        }
    } else {
        if (tetris->state.pending_attack) {
            int p = tetris->state.pending_attack;
            if (p > 8) p = 8;
            tetris->state.pending_attack -= p;
            tetris_receive_garbage_line(tetris, p);
        }
    }
    tetris->state.is_game_over = tetris->game.is_game_over;
}

void tetris_send_garbage_line(Tetris* tetris, int line_count) {
    tetris_receive_garbage_line(tetris, line_count); // self for test?
}