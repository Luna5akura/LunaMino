// core/tetris/tetris.c

#include <raylib.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "tetris_ui/tetris_ui.h"
#include "tetris.h"
#include "tetris_history/tetris_history.h"

#define INPUT_LEFT KEY_LEFT
#define INPUT_RIGHT KEY_RIGHT
#define INPUT_SOFT_DROP KEY_DOWN
#define INPUT_HARD_DROP KEY_SPACE
#define INPUT_ROTATE_CW KEY_X
#define INPUT_ROTATE_CCW KEY_Z
#define INPUT_ROTATE_180 KEY_A
#define INPUT_HOLD KEY_C
#define INPUT_RESTART KEY_R
#define INPUT_UNDO KEY_Q

#define TETRIS_FPS 60
#define TETRIS_GRAVITY 1.0f / 60.0f
#define TETRIS_DAS 100.0f
#define TETRIS_ARR 0.0f
#define TETRIS_SOFT_DROP_GRAVITY 20.0f
#define TETRIS_UNDO_INTERVAL 150.0f
#define TETRIS_LOCK_DELAY 500.0f
#define TETRIS_RESET_LOCK_TIMES_LIMIT 15


const int ATK_TABLE[21][9] = {
    {0, 1, 2, 4, 0, 2, 1, 4, 6},
    {0, 1, 2, 5, 0, 2, 1, 5, 7},
    {1, 1, 3, 6, 1, 3, 1, 6, 9},
    {1, 1, 3, 7, 1, 3, 1, 7, 10},
    {1, 2, 4, 8, 1, 4, 2, 8, 12},
    {1, 2, 4, 9, 1, 4, 2, 9, 13},
    {2, 2, 5, 10, 2, 5, 2, 10, 15},
    {2, 2, 5, 11, 2, 5, 2, 11, 16},
    {2, 3, 6, 12, 2, 6, 3, 12, 18},
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

TetrisConfig* init_tetris_config(Game* game) {
    TetrisConfig* config = (TetrisConfig*)malloc(sizeof(TetrisConfig));

    config->fps = TETRIS_FPS;
    config->gravity = TETRIS_GRAVITY;
    config->das = TETRIS_DAS;
    config->arr = TETRIS_ARR;
    config->soft_drop_gravity = TETRIS_SOFT_DROP_GRAVITY;
    config->drop_interval = 1.0f / (TETRIS_GRAVITY * TETRIS_FPS);
    config->soft_drop_interval = 1.0f / (TETRIS_SOFT_DROP_GRAVITY * TETRIS_FPS);
    config->undo_interval = TETRIS_UNDO_INTERVAL;
    config->lock_delay = TETRIS_LOCK_DELAY;
    config->reset_lock_times_limit = TETRIS_RESET_LOCK_TIMES_LIMIT;

    return config;
}

void free_tetris_config(TetrisConfig* config) {
    free(config);
}

TetrisConfig* copy_tetris_config(TetrisConfig* config) {
    TetrisConfig* new_config = (TetrisConfig*)malloc(sizeof(TetrisConfig));

    memcpy(new_config, config, sizeof(TetrisConfig));

    return new_config;
}

TetrisState* init_tetris_state(Game* game, TetrisConfig* config) {
    TetrisState* state = (TetrisState*)malloc(sizeof(TetrisState));

    state->drop_timer = 0.0f;
    state->das_timer = 0.0f;
    state->arr_timer = 0.0f;
    state->soft_drop_timer = 0.0f;
    state->undo_timer = 0.0f;
    state->lock_timer = 0.0f;
    state->lock_times_left = config->reset_lock_times_limit;
    state->is_left_pressed = FALSE;
    state->is_right_pressed = FALSE;
    state->is_soft_drop_pressed = FALSE;
    state->is_grounded = FALSE;
    state->is_update_clear_rows_needed = FALSE;
    state->attack_type = ATK_NONE;
    state->is_pc = FALSE;
    state->b2b_count = -1;
    state->atk_count = 0;
    state->pending_attack = 0;
    state->is_game_over = FALSE;

    return state;
}

void free_tetris_state(TetrisState* state) {
    free(state);
}

TetrisState* copy_tetris_state(TetrisState* state) {
    TetrisState* new_state = (TetrisState*)malloc(sizeof(TetrisState));

    memcpy(new_state, state, sizeof(TetrisState));

    return new_state;
}

Tetris* init_tetris(Game* game) {
    Tetris* tetris = (Tetris*)malloc(sizeof(Tetris));

    tetris->config = init_tetris_config(game);
    tetris->state = init_tetris_state(game, tetris->config);
    tetris->game = game;

    return tetris;
}

void free_tetris(Tetris* tetris) {
    free_tetris_config(tetris->config);
    free_tetris_state(tetris->state);
    free_game(tetris->game);
    free(tetris);
}

Tetris* copy_tetris(Tetris* tetris) {    
    Tetris* new_tetris = (Tetris*)malloc(sizeof(Tetris));

    new_tetris->config = copy_tetris_config(tetris->config);
    new_tetris->state = copy_tetris_state(tetris->state);
    new_tetris->game = copy_game(tetris->game);

    return new_tetris;
}

void reset_lock_times_left(Tetris* tetris) {
    tetris->state->lock_times_left = tetris->config->reset_lock_times_limit;
    tetris->state->lock_timer = 0.0f;
}

void flush_lock_timer(Tetris* tetris) {
    tetris->state->lock_timer = 0.0f;
    tetris->state->lock_times_left -= 1;
    if (tetris->state->lock_times_left == -1) {
        reset_lock_times_left(tetris);
        tetris->state->is_update_clear_rows_needed = TRUE;
        tetris->state->is_grounded = FALSE;
    }
}

void update_drop_timer(Tetris* tetris) {
    Game* game = tetris->game;

    if (tetris->state->is_grounded) {
        tetris->state->lock_timer += GetFrameTime();
        if (tetris->state->lock_timer * 1000 >= tetris->config->lock_delay) {
            reset_lock_times_left(tetris);
            tetris->state->is_update_clear_rows_needed = TRUE;
            tetris->state->is_grounded = FALSE;
        }
    }
    else {
        tetris->state->drop_timer += GetFrameTime();
        if (tetris->state->drop_timer < tetris->config->drop_interval) return;
        tetris->state->drop_timer = 0.0f;
        try_move_piece(game, MOVE_DOWN);
        tetris->state->is_grounded = is_grounded(game);
    }
}

Tetris* restart_game(Tetris* tetris) {
    free_tetris(tetris);
    Game *game = init_game(TRUE);
    tetris = init_tetris(game);
    return tetris;
}

void detect_left_or_right(Tetris* tetris) {
    Game* game = tetris->game;

    if (IsKeyPressed(INPUT_LEFT) || IsKeyPressed(INPUT_RIGHT)) {
        tetris->state->das_timer = 0;

        Bool is_successful = FALSE;
        if (IsKeyPressed(INPUT_LEFT)) is_successful = try_move_piece(game, MOVE_LEFT);
        if (IsKeyPressed(INPUT_RIGHT)) is_successful = try_move_piece(game, MOVE_RIGHT);
        if (!is_successful) return;

        tetris->state->is_grounded = is_grounded(game);
        if (tetris->state->is_grounded) flush_lock_timer(tetris);
    }
    else {
        tetris->state->das_timer += GetFrameTime() * 1000;
        if (tetris->state->das_timer < tetris->config->das) return;
        // begin arr
        tetris->state->arr_timer += GetFrameTime() * 1000;

        Bool is_looping = TRUE;
        int move_count = -1;

        while (tetris->state->arr_timer >= tetris->config->arr && is_looping) {
            if (IsKeyDown(INPUT_LEFT)) is_looping = try_move_piece(game, MOVE_LEFT);
            if (IsKeyDown(INPUT_RIGHT)) is_looping = try_move_piece(game, MOVE_RIGHT);
            move_count++;
            tetris->state->arr_timer -= tetris->config->arr;
        }
        tetris->state->is_grounded = is_grounded(game);
        if (tetris->state->is_grounded && move_count > 0) flush_lock_timer(tetris);
    }
}

void detect_soft_drop(Tetris* tetris) {
    Game* game = tetris->game;

    if (IsKeyPressed(INPUT_SOFT_DROP)) {
        try_move_piece(game, MOVE_SOFT_DROP);
        tetris->state->soft_drop_timer = 0.0f;
    }
    else {
        tetris->state->soft_drop_timer += GetFrameTime() * 1000;

        Bool is_looping = TRUE;
        while (tetris->state->soft_drop_timer >= tetris->config->soft_drop_gravity && is_looping) {
            is_looping = try_move_piece(game, MOVE_SOFT_DROP);
            tetris->state->soft_drop_timer -= tetris->config->soft_drop_interval;
        }
    }
}

void detect_rotate(Tetris* tetris, RotationAction action) {
    Game* game = tetris->game;

    Bool is_successful = try_rotate_piece(game, action);
    if (!is_successful) return;
    
    tetris->state->is_grounded = is_grounded(game);
    if (tetris->state->is_grounded) flush_lock_timer(tetris);
}

void detect_hold(Tetris* tetris) {
    Game* game = tetris->game;

    tetris->state->is_game_over = try_hold_piece(game);
    tetris->state->drop_timer = 0.0f;
    tetris->state->is_grounded = FALSE;
    reset_lock_times_left(tetris);
}

void detect_hard_drop(Tetris* tetris) {
    Game* game = tetris->game;

    try_move_piece(game, MOVE_HARD_DROP);
    tetris->state->drop_timer = 0.0f;
    tetris->state->is_grounded = FALSE;
    reset_lock_times_left(tetris);

    tetris->state->is_update_clear_rows_needed = TRUE;
}

Tetris* detect_undo(Tetris* tetris, TetrisHistory* tetris_history) {
    if (tetris->state->undo_timer * 1000 < tetris->config->undo_interval)return tetris;
    tetris->state->undo_timer = 0.0f;
    Tetris* pop_tetris = pop_history(tetris_history);
    if (pop_tetris == NULL) return tetris;
    pop_tetris->state->undo_timer = 0.0f;
    free_tetris(tetris);
    return pop_tetris;
}

Tetris* detect_input(Tetris* tetris, TetrisHistory* tetris_history) {
    if (IsKeyDown(INPUT_LEFT) || IsKeyDown(INPUT_RIGHT)) detect_left_or_right(tetris);
    else tetris->state->das_timer = 0.0f;
    if (IsKeyDown(INPUT_SOFT_DROP)) detect_soft_drop(tetris);
    if (IsKeyPressed(INPUT_ROTATE_CCW)) detect_rotate(tetris, ROTATE_CCW);
    if (IsKeyPressed(INPUT_ROTATE_CW)) detect_rotate(tetris, ROTATE_CW);
    if (IsKeyPressed(INPUT_ROTATE_180)) detect_rotate(tetris, ROTATE_180);
    if (IsKeyPressed(INPUT_HOLD)) detect_hold(tetris);
    if (IsKeyPressed(INPUT_HARD_DROP)) detect_hard_drop(tetris);
    if (IsKeyDown(INPUT_RESTART)) tetris = restart_game(tetris);
    if (IsKeyDown(INPUT_UNDO)) tetris = detect_undo(tetris, tetris_history);
    return tetris;
}

int get_s2_atk(Tetris* tetris) {
    printf("attack type: %d\n", tetris->state->attack_type);

    int total_atk = 0;
    int b2b_charging = 0;
    if (
        tetris->state->b2b_count > 4
        && (
            tetris->state->attack_type == ATK_SINGLE
            || tetris->state->attack_type == ATK_DOUBLE
            || tetris->state->attack_type == ATK_TRIPLE
        )
    ) {
        // b2b charging
        b2b_charging = tetris->state->b2b_count;
    }

    if (
        tetris->state->b2b_count > 0
        && tetris->state->attack_type != ATK_SINGLE
        && tetris->state->attack_type != ATK_DOUBLE
        && tetris->state->attack_type != ATK_TRIPLE
    ) {
        // b2b attack
        int ren = tetris->game->state->ren;
        ren = ren != -1 ? ren : 0;
        int atk_type = (int)tetris->state->attack_type;
        atk_type = atk_type <= (int)ATK_TST ? atk_type : (int)ATK_TSMS;
        total_atk += ATK_TABLE_B2B1[ren][atk_type - (int)ATK_TETRIS];
    }
    else {
        // no b2b attack
        int ren = tetris->game->state->ren;
        ren = ren != -1 ? ren : 0;
        int atk_type = (int)tetris->state->attack_type;
        atk_type = atk_type <= (int)ATK_TST ? atk_type : (int)ATK_TSMS;
        total_atk += ATK_TABLE[ren][atk_type - (int)ATK_SINGLE];
    }

    if (tetris->state->is_pc == TRUE) {
        // perfect clear
        total_atk += 5;
        if (
            tetris->state->attack_type == ATK_SINGLE
            || tetris->state->attack_type == ATK_DOUBLE
            || tetris->state->attack_type == ATK_TRIPLE
        ) {
            tetris->state->b2b_count += 1;
        }
    }

    total_atk += b2b_charging;
    return total_atk;
}

int get_atk(Tetris* tetris) {
    if (tetris->state->attack_type == ATK_NONE) return;
    return get_s2_atk(tetris);
}



void receive_garbage_line(Tetris* tetris, int line_count) {
    Board* board = tetris->game->board;
    int hole_column = random() % board->width;

    for (int row = 0; row < board->height - line_count; row++) {
        for (int col = 0; col < board->width; col++) {
            board->state[col][row + line_count] = board->state[col][row];
        }
    }

    for (int row = 0; row < line_count; row++) {
        for (int col = 0; col < board->width; col++) {
            if (col == hole_column) {
                board->state[col][row] = 0;
            }
            else {
                board->state[col][row] = 8;
            }
        }
    }   
}

void send_garbage_line(Tetris* tetris, int line_count) {
    receive_garbage_line(tetris, line_count);
}

void receive_attack(Tetris* tetris, int attack) {
    tetris->state->pending_attack += attack;
}

void update_clear_rows(Tetris* tetris) {
    int attack;
    tetris->state->drop_timer = 0.0f;
    tetris->state->is_update_clear_rows_needed = FALSE;

    Game* game = tetris->game;
    printf("update_clear_rows_needed\n");
    
    tetris->state->attack_type = get_attack_type(game);
    tetris->state->is_pc = is_perfect_clear(game);
    update_ren(game);
    attack = get_atk(tetris);
    tetris->state->atk_count = attack;
    tetris->state->is_game_over = next_piece(game);
    int clear_count = clear_rows(game->board);
    if (
        tetris->state->attack_type == ATK_SINGLE 
        || tetris->state->attack_type == ATK_DOUBLE
        || tetris->state->attack_type == ATK_TRIPLE
    ) tetris->state->b2b_count = -1;
    else tetris->state->b2b_count++;

    if (clear_count == 0) {
        if (tetris->state->pending_attack == 0) return;
        if (tetris->state->pending_attack > 8) {
            tetris->state->pending_attack -= 8;
            receive_garbage_line(tetris, 8);  
        }
        else {
            tetris->state->pending_attack = 0;
            receive_garbage_line(tetris, tetris->state->pending_attack);
        }
    }
    else {
        if (tetris->state->pending_attack == 0) {
            send_garbage_line(tetris, attack);
        }
        else {
            if (attack > tetris->state->pending_attack) {
                send_garbage_line(tetris, attack - tetris->state->pending_attack);
                tetris->state->pending_attack = 0;
            }
            else {
                tetris->state->pending_attack -= attack;
            }
        }
    }
}

void run_game(Game* game) {
    Tetris* tetris = init_tetris(game);
    UIConfig* ui_config = init_ui_config();
    TetrisHistory* tetris_history = init_tetris_history(300);

    init_window(tetris, ui_config);
    SetTargetFPS(tetris->config->fps);
    while (!WindowShouldClose()) {
        tetris = detect_input(tetris, tetris_history);
        
        update_drop_timer(tetris);
        tetris->state->undo_timer += GetFrameTime();
        if (tetris->state->is_update_clear_rows_needed) {
            update_clear_rows(tetris);
            push_history(tetris_history, tetris);
        }
        draw_content(tetris, ui_config);
    }
}

int main() {
    Game* game = init_game(TRUE);
    run_game(game);
    free_game(game);
    CloseWindow();
    return 0;
}