// core/tetris/tetris.c

#ifndef SHARED_LIB
    #include <raylib.h>
    #include "tetris_ui/tetris_ui.h"
    #include "tetris_history/tetris_history.h"
    #include "tetris.h"
#else
    #include "tetris.h"
    #include "../game/game.h"
    
    // 即使是共享库，为了可视化也需要包含 raylib 和 ui
    #include <raylib.h>
    #include "tetris_ui/tetris_ui.h" 
#endif

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define TETRIS_FPS 60
#define TETRIS_GRAVITY 1.0f / 60.0f
#define TETRIS_DAS 100.0f
#define TETRIS_ARR 0.0f
#define TETRIS_SOFT_DROP_GRAVITY 0.0f
#define TETRIS_UNDO_INTERVAL 150.0f
#define TETRIS_LOCK_DELAY 500.0f
#define TETRIS_RESET_LOCK_TIMES_LIMIT 15

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
    config->soft_drop_interval = TETRIS_SOFT_DROP_GRAVITY;
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
    if (tetris == NULL) return;
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

// ============================================================================
// UI / Control Logic (Non-Shared Only)
// ============================================================================
#ifndef SHARED_LIB
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
    Game *game = init_game();
    tetris = init_tetris(game);
    return tetris;
}

void detect_left_or_right(Tetris* tetris, InputControl controls) {
    Game* game = tetris->game;
    if (IsKeyPressed(controls.left) || IsKeyPressed(controls.right)) {
        tetris->state->das_timer = 0;
        Bool is_successful = FALSE;
        if (IsKeyPressed(controls.left)) is_successful = try_move_piece(game, MOVE_LEFT);
        if (IsKeyPressed(controls.right)) is_successful = try_move_piece(game, MOVE_RIGHT);
        if (!is_successful) return;
        tetris->state->is_grounded = is_grounded(game);
        if (tetris->state->is_grounded) flush_lock_timer(tetris);
    }
    else {
        tetris->state->das_timer += GetFrameTime() * 1000;
        if (tetris->state->das_timer < tetris->config->das) return;
        tetris->state->arr_timer += GetFrameTime() * 1000;
        Bool is_looping = TRUE;
        int move_count = -1;
        while (tetris->state->arr_timer >= tetris->config->arr && is_looping) {
            if (IsKeyDown(controls.left)) is_looping = try_move_piece(game, MOVE_LEFT);
            if (IsKeyDown(controls.right)) is_looping = try_move_piece(game, MOVE_RIGHT);
            move_count++;
            tetris->state->arr_timer -= tetris->config->arr;
        }
        tetris->state->is_grounded = is_grounded(game);
        if (tetris->state->is_grounded && move_count > 0) flush_lock_timer(tetris);
    }
}

void detect_soft_drop(Tetris* tetris, int key_soft_drop) {
    Game* game = tetris->game;
    if (IsKeyDown(key_soft_drop)) {
        if (tetris->config->soft_drop_gravity <= 0.0f) {
            while (try_move_piece(game, MOVE_SOFT_DROP));
            tetris->state->soft_drop_timer = 0.0f;
            return;
        }
        if (IsKeyPressed(key_soft_drop)) {
            try_move_piece(game, MOVE_SOFT_DROP);
            tetris->state->soft_drop_timer = 0.0f;
        }
        else {
            tetris->state->soft_drop_timer += GetFrameTime() * 1000;
            Bool is_looping = TRUE;
            float interval = tetris->config->soft_drop_interval;
            if (interval <= 0.0001f) interval = 1.0f;
            while (tetris->state->soft_drop_timer >= tetris->config->soft_drop_gravity && is_looping) {
                is_looping = try_move_piece(game, MOVE_SOFT_DROP);
                tetris->state->soft_drop_timer -= interval;
            }
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
    if (tetris->state->undo_timer * 1000 < tetris->config->undo_interval) return tetris;
    tetris->state->undo_timer = 0.0f;
    Tetris* pop_tetris = pop_history(tetris_history);
    if (pop_tetris == NULL) return tetris;
    pop_tetris->state->undo_timer = 0.0f;
    free_tetris(tetris);
    return pop_tetris;
}

Tetris* detect_input(Tetris* tetris, void* tetris_history_ptr, InputControl controls) {
    TetrisHistory* tetris_history = (TetrisHistory*)tetris_history_ptr;
    if (IsKeyDown(controls.left) || IsKeyDown(controls.right)) detect_left_or_right(tetris, controls);
    else tetris->state->das_timer = 0.0f;
   
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
    AttackType type = tetris->state->attack_type;
    int ren = tetris->game->state->ren;
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
        if (tetris->state->b2b_count > 0 && keeps_b2b) {
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
        if (tetris->state->b2b_count > 0) base_dmg += 1;
    }

    if (!keeps_b2b && tetris->state->b2b_count > 4) {
        total_atk += tetris->state->b2b_count;
    }

    if (tetris->state->is_pc) {
        total_atk += 10;
    }
    total_atk += base_dmg;
    return total_atk;
}

int get_atk(Tetris* tetris) {
    if (tetris->state->attack_type == ATK_NONE) return 0;
    return get_s2_atk(tetris);
}

void receive_garbage_line(Tetris* tetris, int line_count) {
    Board* board = tetris->game->board;
    int hole_column = random() % board->width;
    for (int row = board->height - line_count - 1; row >= 0; row--) {
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
   
    tetris->state->attack_type = get_attack_type(game);
    tetris->state->is_pc = is_perfect_clear(game);
    update_ren(game);
    attack = get_atk(tetris);
    tetris->state->atk_count = attack;
    tetris->state->is_game_over = next_piece(game);
    int clear_count = clear_rows(game->board);
   
    if (tetris->state->attack_type == ATK_SINGLE ||
        tetris->state->attack_type == ATK_DOUBLE ||
        tetris->state->attack_type == ATK_TRIPLE) {
        tetris->state->b2b_count = -1;
    } else if (tetris->state->attack_type != ATK_NONE) {
        tetris->state->b2b_count++;
    }
    if (clear_count == 0) {
        if (tetris->state->pending_attack == 0) return;
        if (tetris->state->pending_attack > 8) {
            tetris->state->pending_attack -= 8;
            receive_garbage_line(tetris, 8);
        }
        else {
            int p = tetris->state->pending_attack;
            tetris->state->pending_attack = 0;
            receive_garbage_line(tetris, p);
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

#ifndef SHARED_LIB
void run_game(Game* game) {
    Tetris* tetris = init_tetris(game);
    UIConfig* ui_config = init_ui_config();
    TetrisHistory* tetris_history = init_tetris_history(300);
    init_window(tetris, ui_config);
    SetTargetFPS(tetris->config->fps);
   
    InputControl controls_default = {
        .left = KEY_LEFT, .right = KEY_RIGHT, .down = KEY_DOWN,
        .hard_drop = KEY_SPACE,
        .rotate_cw = KEY_X, .rotate_ccw = KEY_Z, .rotate_180 = KEY_A,
        .hold = KEY_C, .undo = KEY_Q, .restart = KEY_R
    };
    while (!WindowShouldClose()) {
        tetris = detect_input(tetris, tetris_history, controls_default);
       
        update_drop_timer(tetris);
        tetris->state->undo_timer += GetFrameTime();
        if (tetris->state->is_update_clear_rows_needed) {
            update_clear_rows(tetris);
            push_history(tetris_history, tetris);
        }
        draw_content(tetris, ui_config);
    }
   
    free_tetris_history(tetris_history);
    free_tetris(tetris);
    free_ui_config(ui_config);
}
#endif 

// ============================================================================
// AI Implementation (Exposed Functions)
// ============================================================================
typedef struct {
    int x, y, rot;
    int shape[4][4];
} BFSNode;

typedef struct {
    BFSNode data[2048]; 
    int head;
    int tail;
} BFSQueue;

void bfs_push(BFSQueue* q, int x, int y, int rot, int shape[4][4]) {
    if (q->tail >= 2048) return;
    q->data[q->tail].x = x;
    q->data[q->tail].y = y;
    q->data[q->tail].rot = rot;
    memcpy(q->data[q->tail].shape, shape, sizeof(int) * 16);
    q->tail++;
}

BFSNode bfs_pop(BFSQueue* q) {
    return q->data[q->head++];
}

int bfs_empty(BFSQueue* q) {
    return q->head == q->tail;
}

void run_bfs_for_current_piece(Game* game, int use_hold_flag, LegalMoves* out_moves, int* move_count) {
    Piece* p = game->current_piece;
    Board* board = game->board;
    
    static char visited[4][12][40]; 
    memset(visited, 0, sizeof(visited));

    BFSQueue q;
    q.head = 0; q.tail = 0;

    bfs_push(&q, p->x, p->y, (int)p->rotation, p->shape);
    visited[(int)p->rotation][p->x][p->y] = 1;

    Piece original_piece = *p;
    
    while (!bfs_empty(&q)) {
        BFSNode curr = bfs_pop(&q);
        
        p->x = curr.x;
        p->y = curr.y;
        p->rotation = (Rotation)curr.rot;
        memcpy(p->shape, curr.shape, sizeof(p->shape));

        int current_y = p->y;
        while (!is_overlapping(board, p)) p->y--;
        p->y++; 
        int ghost_y = p->y;
        p->y = current_y;

        Bool exists = FALSE;
        for(int i = 0; i < *move_count; i++) {
            if (out_moves->moves[i].x == p->x && 
                out_moves->moves[i].y == ghost_y && 
                out_moves->moves[i].rotation == (int)p->rotation &&
                out_moves->moves[i].use_hold == use_hold_flag) {
                exists = TRUE;
                break;
            }
        }
        if (!exists && *move_count < MAX_LEGAL_MOVES) {
            out_moves->moves[*move_count].x = p->x;
            out_moves->moves[*move_count].y = ghost_y; 
            out_moves->moves[*move_count].rotation = (int)p->rotation;
            out_moves->moves[*move_count].use_hold = use_hold_flag;
            (*move_count)++;
        }

        for (int action = 0; action < 5; action++) {
            p->x = curr.x; p->y = curr.y; p->rotation = (Rotation)curr.rot;
            memcpy(p->shape, curr.shape, sizeof(p->shape));

            Bool success = FALSE;
            if (action == 0) success = try_move_piece(game, MOVE_LEFT);
            else if (action == 1) success = try_move_piece(game, MOVE_RIGHT);
            else if (action == 2) success = try_move_piece(game, MOVE_DOWN);
            else if (action == 3) success = try_rotate_piece(game, ROTATE_CW);
            else if (action == 4) success = try_rotate_piece(game, ROTATE_CCW);
            
            if (success) {
                int nx = p->x; int ny = p->y; int nr = (int)p->rotation;
                if (nx >= 0 && nx < 12 && ny >= 0 && ny < 40) {
                    if (!visited[nr][nx][ny]) {
                        visited[nr][nx][ny] = 1;
                        bfs_push(&q, nx, ny, nr, p->shape);
                    }
                }
            }
        }
    }
    *p = original_piece;
}

void ai_reset_game(Tetris* tetris, int seed) {
    if (tetris == NULL) return;
    if (tetris->game != NULL) {
        free_game(tetris->game);
        tetris->game = NULL;
    }
    
    GameConfig* config = init_game_config();
    config->seed = seed;
    srandom(seed);
    
    tetris->game = malloc(sizeof(Game));
    tetris->game->config = config;
    tetris->game->state = init_game_state(config);
    tetris->game->board = init_board();
    
    Piece* current_piece = init_piece(bag_next_piece(tetris->game->state->bag));
    current_piece->x = 3;
    current_piece->y = 21;
    tetris->game->current_piece = current_piece;

    tetris->state->b2b_count = -1;
    tetris->state->atk_count = 0;
    tetris->state->pending_attack = 0;
    tetris->state->is_game_over = FALSE;
    tetris->game->state->ren = -1;
    tetris->state->lock_timer = 0;
    tetris->state->drop_timer = 0;
}

void ai_get_state(Tetris* tetris, int* board_buffer, int* queue_buffer, int* hold_buffer, int* meta_buffer) {
    Game* game = tetris->game;
    int idx = 0;
    for (int y = 0; y < 20; y++) {
        for (int x = 0; x < 10; x++) {
            board_buffer[idx++] = (game->board->state[x][y] > 0 ? 1 : 0); 
        }
    }
    for (int i = 0; i < 5; i++) {
        queue_buffer[i] = (int)game->state->previews->previews[i];
    }
    if (game->state->hold_piece) hold_buffer[0] = (int)game->state->hold_piece->type + 1; 
    else hold_buffer[0] = 0;
    meta_buffer[0] = tetris->state->b2b_count;
    meta_buffer[1] = game->state->ren;
    meta_buffer[2] = game->state->can_hold_piece;
    meta_buffer[3] = (int)game->current_piece->type;
}

void ai_get_legal_moves(Tetris* tetris, LegalMoves* out_moves) {
    out_moves->count = 0;
    Game* game = tetris->game;
    
    Piece original_piece_backup = *game->current_piece;
    int original_can_hold = game->state->can_hold_piece;
    
    // Pass 1: Current
    run_bfs_for_current_piece(game, 0, out_moves, &(out_moves->count));
    
    // Pass 2: Hold
    if (game->config->is_hold_enabled && game->state->can_hold_piece) {
        Piece* old_current = game->current_piece;
        Piece* new_current = NULL;
        
        if (game->state->hold_piece == NULL) {
            // Hold为空时，取 Preview 队列的当前头部
            // 【修复】直接使用 previews[current]
            Previews* p = game->state->previews;
            new_current = init_piece(p->previews[p->current]);
        } else {
            new_current = init_piece(game->state->hold_piece->type);
        }
        
        if (new_current != NULL) {
            new_current->x = 3;
            new_current->y = 21;
            game->current_piece = new_current;
            run_bfs_for_current_piece(game, 1, out_moves, &(out_moves->count));
            free_piece(new_current);
        }
        
        game->current_piece = old_current;
        *game->current_piece = original_piece_backup;
    }
    game->state->can_hold_piece = original_can_hold;
}

StepResult ai_step(Tetris* tetris, int x, int rotation, int use_hold) {
    StepResult result;
    memset(&result, 0, sizeof(StepResult));
    Game* game = tetris->game;
    
    if (tetris->state->is_game_over) {
        result.is_game_over = TRUE;
        return result;
    }

    // 1. 处理 Hold
    if (use_hold) try_hold_piece(game);

    Piece* p = game->current_piece;
    
    // 2. 强制设置旋转
    Piece* temp = init_piece(p->type);
    memcpy(p->shape, temp->shape, sizeof(p->shape));
    free_piece(temp);
    p->rotation = (Rotation)0;
    while ((int)p->rotation != rotation) rotate_piece(p, ROTATE_CW);
    
    // 3. 设置 X 坐标
    p->x = x;
    
    // 4. 智能瞬移 (寻找最低落点)
    int best_y = -999;
    for (int y = -4; y < 40; y++) {
        p->y = y;
        if (!is_overlapping(game->board, p)) {
            p->y = y - 1;
            Bool is_ground = is_overlapping(game->board, p);
            p->y = y;
            
            if (is_ground) {
                best_y = y;
                break;
            }
        }
    }
    
    if (best_y != -999) {
        p->y = best_y;
    } else {
        p->y = 20; 
    }

    // 5. T-Spin 标记 Hack
    if (p->type == T_PIECE) {
        game->state->is_last_rotate = 1; 
    } else {
        game->state->is_last_rotate = 0;
    }
    
    // 6. 计算攻击信息 (必须在 next_piece 之前做，因为依靠 current_piece)
    tetris->state->attack_type = get_attack_type(game);
    tetris->state->is_pc = is_perfect_clear(game);
    update_ren(game);
    int atk = get_atk(tetris); // 计算本次攻击力
    
    result.attack_type = tetris->state->attack_type;
    result.combo_count = game->state->ren; // 记录当前的连击数
    
    // B2B 状态更新 (仅更新状态，不决定是否发送垃圾，因为还要看是否消行)
    if (result.attack_type != ATK_NONE && result.attack_type != ATK_SINGLE && 
        result.attack_type != ATK_DOUBLE && result.attack_type != ATK_TRIPLE) {
        tetris->state->b2b_count++;
        result.b2b_count = tetris->state->b2b_count;
    } else if (result.attack_type != ATK_NONE) {
        tetris->state->b2b_count = -1;
        result.b2b_count = -1;
    } else {
        result.b2b_count = tetris->state->b2b_count;
    }
    
    // =================================================================
    // 【关键修改】 顺序调整
    // 1. 先调用 next_piece 将当前方块锁定(Lock)到棋盘上
    // 2. 再调用 clear_rows 检查是否有满行
    // =================================================================
    
    // 锁定并生成下一个 (Lock & Spawn)
    result.is_game_over = (int)next_piece(game); 

    // 此时棋盘 board->state 已经更新，可以正确检测消行了
    result.lines_cleared = clear_rows(game->board);
    
    // 7. 垃圾行结算 (基于正确的 lines_cleared)
    result.damage_sent = 0;
    if (result.lines_cleared > 0) {
        // 这一步消行了
        if (tetris->state->pending_attack > 0) {
            // 抵消垃圾
            if (atk >= tetris->state->pending_attack) {
                result.damage_sent = atk - tetris->state->pending_attack;
                tetris->state->pending_attack = 0;
            } else {
                tetris->state->pending_attack -= atk;
                result.damage_sent = 0;
            }
        } else {
            // 发送攻击
            result.damage_sent = atk;
        }
    } else {
        // 没有消行，接收垃圾
        if (tetris->state->pending_attack > 0) {
            int trash = tetris->state->pending_attack;
            if (trash > 8) trash = 8; // 每次最多接8行 (根据你的规则)
            tetris->state->pending_attack -= trash;
            receive_garbage_line(tetris, trash);
        }
    }
    
    return result;
}



void ai_receive_garbage(Tetris* tetris, int lines) {
    receive_attack(tetris, lines);
}

static UIConfig* ai_ui_config = NULL;

void ai_enable_visualization(Tetris* tetris) {
    if (ai_ui_config == NULL) {
        SetTraceLogLevel(LOG_WARNING); 
        ai_ui_config = init_ui_config();
        init_window(tetris, ai_ui_config); 
        SetTargetFPS(60);
    }
}

void ai_render(Tetris* tetris) {
    if (ai_ui_config != NULL && !WindowShouldClose()) {
        draw_content(tetris, ai_ui_config);
        PollInputEvents();
    }
}

void ai_close_visualization() {
    if (ai_ui_config != NULL) {
        CloseWindow(); 
        free_ui_config(ai_ui_config);
        ai_ui_config = NULL;
    }
}