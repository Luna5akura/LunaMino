// core/interact/interact.c

#include <raylib.h>
#include <stdlib.h>
#include <stdio.h>
#include "interact.h"

#define INPUT_LEFT KEY_LEFT
#define INPUT_RIGHT KEY_RIGHT
#define INPUT_SOFT_DROP KEY_DOWN
#define INPUT_HARD_DROP KEY_SPACE
#define INPUT_ROTATE_CW KEY_X
#define INPUT_ROTATE_CCW KEY_Z
#define INPUT_ROTATE_180 KEY_A
#define INPUT_HOLD KEY_C
#define INPUT_RESTART KEY_R

#define TETRIS_GAME_HEIGHT 20

#define TETRIS_WIDTH 800
#define TETRIS_HEIGHT 600
#define TETRIS_BLOCK_SIZE 20
#define TETRIS_IS_SHADOW_ENABLED TRUE

#define TETRIS_FPS 60
#define TETRIS_GRAVITY 1.0f / 60.0f
#define TETRIS_DAS 100.0f
#define TETRIS_ARR 0.0f
#define TETRIS_SOFT_DROP_GRAVITY 20.0f
#define TETRIS_LOCK_DELAY 500.0f
#define TETRIS_RESET_LOCK_TIMES_LIMIT 15

const char TETRIS_ATK_STR[28][20] = {
    "",
    "Single",
    "Double",
    "Triple",
    "Quad",
    "T-Spin Single Mini",
    "T-Spin Single",
    "T-Spin Double Mini",
    "T-Spin Double",
    "T-Spin Triple",
    "I-Spin Single",
    "I-Spin Double",
    "I-Spin Triple",
    "O-Spin Single",
    "O-Spin Double",
    "S-Spin Single",
    "S-Spin Double",
    "S-Spin Triple",
    "Z-Spin Single",
    "Z-Spin Double",
    "Z-Spin Triple",
    "L-Spin Single",
    "L-Spin Double",
    "L-Spin Triple",
    "J-Spin Single",
    "J-Spin Double",
    "J-Spin Triple",
    "ERROR",
};

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

Color GetColorForPieceType(int type) {
    switch (type) {
        case 0: return GRAY;        //
        case 1: return SKYBLUE;     // I
        case 2: return YELLOW;      // O
        case 3: return PURPLE;      // T
        case 4: return GREEN;       // S 
        case 5: return RED;         // Z
        case 6: return BLUE;        // J
        case 7: return ORANGE;      // L
        default: return BLACK;
    }
}

void init_window(Tetris* tetris) {
    int screenWidth = tetris->config->width;
    int screenHeight = tetris->config->height;
    InitWindow(screenWidth, screenHeight, "Tetris Game");
}

void draw_game_over(Tetris* tetris) {
    DrawText(
        "Game Over", 
        tetris->config->width / 2 - MeasureText("Game Over", 40) / 2, 
        tetris->config->height / 2 - 20, 
        40, 
        RED
    );
}

void exit_window() {}

TetrisConfig* init_tetris_config(Game* game) {
    TetrisConfig* config = (TetrisConfig*)malloc(sizeof(TetrisConfig));

    config->width = TETRIS_WIDTH;       
    config->height = TETRIS_HEIGHT;     
    config->block_size = TETRIS_BLOCK_SIZE;
    config->is_shadow_enabled = TETRIS_IS_SHADOW_ENABLED;

    config->fps = TETRIS_FPS;
    config->gravity = TETRIS_GRAVITY;
    config->das = TETRIS_DAS;
    config->arr = TETRIS_ARR;
    config->soft_drop_gravity = TETRIS_SOFT_DROP_GRAVITY;
    config->drop_interval = 1.0f / (TETRIS_GRAVITY * TETRIS_FPS);
    config->soft_drop_interval = 1.0f / (TETRIS_SOFT_DROP_GRAVITY * TETRIS_FPS);
    config->lock_delay = TETRIS_LOCK_DELAY;
    config->reset_lock_times_limit = TETRIS_RESET_LOCK_TIMES_LIMIT;

    return config;
}

void free_tetris_config(TetrisConfig* config) {
    free(config);
}

TetrisState* init_tetris_state(Game* game, TetrisConfig* config) {
    TetrisState* state = (TetrisState*)malloc(sizeof(TetrisState));

    state->drop_timer = 0.0f;
    state->das_timer = 0.0f;
    state->arr_timer = 0.0f;
    state->soft_drop_timer = 0.0f;
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
    state->is_game_over = FALSE;

    return state;
}

void free_tetris_state(TetrisState* state) {
    free(state);
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

void restart_game(Tetris* tetris) {
    free_tetris(tetris);
    Game *game = init_game(TRUE);
    tetris = init_tetris(game);
}

void detect_left_or_right(Tetris* tetris) {
    Game* game = tetris->game;

    if (IsKeyPressed(INPUT_LEFT) || IsKeyPressed(INPUT_RIGHT)) {
        tetris->state->das_timer = 0;

        Bool is_successful = FALSE;
        if (IsKeyPressed(INPUT_LEFT)) is_successful = try_move_piece(game, MOVE_LEFT);
        if (IsKeyPressed(INPUT_RIGHT)) is_successful = try_move_piece(game, MOVE_RIGHT);
        if (!is_successful) return;

        if (tetris->state->is_grounded) flush_lock_timer(tetris);
        else tetris->state->lock_timer = 0.0f;

        tetris->state->is_grounded = is_grounded(game);
    }
    else {
        tetris->state->das_timer += GetFrameTime() * 1000;
        if (!tetris->state->das_timer >= tetris->config->das) return;
        // begin arr
        tetris->state->arr_timer += GetFrameTime() * 1000;

        Bool is_looping = TRUE;
        int move_count = -1;

        while (tetris->state->arr_timer >= tetris->config->arr && is_looping) {
            if (IsKeyDown(KEY_LEFT)) is_looping = try_move_piece(game, MOVE_LEFT);
            if (IsKeyDown(KEY_RIGHT)) is_looping = try_move_piece(game, MOVE_RIGHT);
            move_count++;
            tetris->state->arr_timer -= tetris->config->arr;
        }
        if (tetris->state->is_grounded && move_count > 0) flush_lock_timer(tetris);
        tetris->state->is_grounded = is_grounded(game);
    }
}

void detect_soft_drop(Tetris* tetris) {
    Game* game = tetris->game;

    if (IsKeyPressed(KEY_DOWN)) {
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
    
    if (tetris->state->is_grounded) flush_lock_timer(tetris);
    else tetris->state->lock_timer = 0.0f;

    tetris->state->is_grounded = is_grounded(game);
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

    tetris->state->is_game_over = try_move_piece(game, MOVE_HARD_DROP);
    tetris->state->drop_timer = 0.0f;
    tetris->state->is_grounded = FALSE;
    reset_lock_times_left(tetris);

    tetris->state->is_update_clear_rows_needed = TRUE;
}


void detect_input(Tetris* tetris) {
    Game* game = tetris->game;

    if (IsKeyDown(INPUT_LEFT) || IsKeyDown(INPUT_RIGHT)) detect_left_or_right(tetris);
    else tetris->state->das_timer = 0.0f;
    if (IsKeyDown(INPUT_SOFT_DROP)) detect_soft_drop(tetris);
    if (IsKeyPressed(INPUT_ROTATE_CCW)) detect_rotate(tetris, ROTATE_CCW);
    if (IsKeyPressed(INPUT_ROTATE_CW)) detect_rotate(tetris, ROTATE_CW);
    if (IsKeyPressed(INPUT_ROTATE_180)) detect_rotate(tetris, ROTATE_180);
    if (IsKeyPressed(INPUT_HOLD)) detect_hold(tetris);
    if (IsKeyPressed(INPUT_HARD_DROP)) detect_hard_drop(tetris);
    if (IsKeyDown(INPUT_RESTART)) restart_game(tetris);
}

int get_s2_atk(Tetris* tetris) {\
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

void update_atk(Tetris* tetris) {
    if (tetris->state->attack_type == ATK_NONE) return;
    tetris->state->atk_count = get_s2_atk(tetris);
}

void draw_board(Tetris* tetris) {
    Game* game = tetris->game;

    int blockSize = tetris->config->block_size;
    int boardOffsetX = (tetris->config->width - game->board->width * blockSize) / 2;
    int boardOffsetY = (tetris->config->height - game->board->height * blockSize) / 2;
    
    for (int y = 0; y < game->board->height; y++) {
        for (int x = 0; x < game->board->width; x++) {
            if (y >= TETRIS_GAME_HEIGHT && game->board->state[x][y] == 0) continue;

            Color color = GetColorForPieceType(game->board->state[x][y]);
            
            DrawRectangle(
                boardOffsetX + x * blockSize, 
                boardOffsetY + (game->board->height - y - 1) * blockSize, 
                blockSize, 
                blockSize, 
                color
            );

            DrawRectangleLines(
                boardOffsetX + x * blockSize, 
                boardOffsetY + (game->board->height - y - 1) * blockSize, 
                blockSize, 
                blockSize, 
                DARKGRAY
            );
        }
    }
}

void draw_piece(Tetris* tetris) {
    Game* game = tetris->game;

    int blockSize = tetris->config->block_size;
    int boardOffsetX = (tetris->config->width - game->board->width * blockSize) / 2;
    int boardOffsetY = (tetris->config->height - game->board->height * blockSize) / 2;    

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (game->current_piece->shape[i][j] == 0) continue;
            int x = game->current_piece->x + j;
            int y = game->current_piece->y - i;
            if (y < 0) continue;

            Color color = GetColorForPieceType(game->current_piece->type + 1);
            DrawRectangle(
                boardOffsetX + x * blockSize, 
                boardOffsetY + (game->board->height - y - 1) * blockSize, 
                blockSize, 
                blockSize, 
                color
            );
            DrawRectangleLines(
                boardOffsetX + x * blockSize, 
                boardOffsetY + (game->board->height - y - 1) * blockSize, 
                blockSize, 
                blockSize, 
                DARKGRAY
            );
        }
    }
}

void draw_previews(Tetris* tetris) {
    Game* game = tetris->game;
    int blockSize = tetris->config->block_size;
    int boardOffsetX = (tetris->config->width - game->board->width * blockSize) / 2;
    int boardOffsetY = (tetris->config->height - game->board->height * blockSize) / 2;    

    int previewOffsetX = boardOffsetX + (game->board->width + 2) * blockSize;
    int previewOffsetY = boardOffsetY;
    for (int p = 0; p < game->config->preview_count; p++) {
        Piece* preview_piece = init_piece(game->state->previews->previews[(p + game->state->previews->current) % 5]);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (preview_piece->shape[i][j] != 0) {
                    Color color = GetColorForPieceType(preview_piece->type + 1);
                    DrawRectangle(
                        previewOffsetX + j * blockSize, 
                        previewOffsetY + p * 4 * blockSize + i * blockSize, 
                        blockSize, 
                        blockSize, 
                        color
                    );
                    DrawRectangleLines(
                        previewOffsetX + j * blockSize, 
                        previewOffsetY + p * 4 * blockSize + i * blockSize, 
                        blockSize, 
                        blockSize, 
                        DARKGRAY
                    );
                }
            }
        }
        free_piece(preview_piece);
    }
}

void draw_hold_piece(Tetris* tetris) {
    Game* game = tetris->game;

    if (game->state->hold_piece == NULL) return;
    
    int blockSize = tetris->config->block_size;

    int boardOffsetX = (tetris->config->width - game->board->width * blockSize) / 2;
    int boardOffsetY = (tetris->config->height - game->board->height * blockSize) / 2;    

    int holdOffsetX = boardOffsetX - (game->board->width + 2) * blockSize / 2;
    int holdOffsetY = boardOffsetY;

    Piece* hold_piece = game->state->hold_piece;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (hold_piece->shape[i][j] != 0) {
                Color color = GetColorForPieceType(hold_piece->type + 1);
                DrawRectangle(
                    holdOffsetX + j * blockSize, 
                    holdOffsetY + i * blockSize, 
                    blockSize, 
                    blockSize, 
                    color
                );
                DrawRectangleLines(
                    holdOffsetX + j * blockSize, 
                    holdOffsetY + i * blockSize, 
                    blockSize, 
                    blockSize, 
                    DARKGRAY
                );
            }
        }
    }
};

void draw_shadow(Tetris* tetris) {
    Game* game = tetris->game;

    int shadow_height = get_shadow_height(game);
    if (shadow_height == 0) return;


    int blockSize = tetris->config->block_size;
    int boardOffsetX = (tetris->config->width - game->board->width * blockSize) / 2;
    int boardOffsetY = (tetris->config->height - game->board->height * blockSize) / 2;    

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (game->current_piece->shape[i][j] == 0) continue;
            int x = game->current_piece->x + j;
            int y = game->current_piece->y - i;
            if (y < 0) continue;

            Color color = GetColorForPieceType(game->current_piece->type + 1);
            color.a /= 3;
            DrawRectangle(
                boardOffsetX + x * blockSize, 
                boardOffsetY + (game->board->height - y - 1 + shadow_height) * blockSize, 
                blockSize, 
                blockSize, 
                color
            );
            DrawRectangleLines(
                boardOffsetX + x * blockSize, 
                boardOffsetY + (game->board->height - y - 1 + shadow_height) * blockSize, 
                blockSize, 
                blockSize, 
                DARKGRAY
            );
        }
    }
}

void draw_debug_info(Tetris* tetris) {
    char buffer[2048];

    snprintf(buffer, 2048,
        "drop_timer: %.2f\n"
        "das_timer: %.2f\n"
        "arr_timer: %.2f\n"
        "soft_drop_timer: %.2f\n"
        "lock_timer: %.2f\n"
        "lock_times_left: %i\n"
        "is_left_pressed: %i\n"
        "is_right_pressed: %i\n"
        "is_soft_drop_pressed: %i\n"
        "is_grounded: %i\n"
        "is_update_clear_rows_needed: %i\n"
        "attack_type: %i\n"
        "is_pc: %i\n"
        "current_piece->x: %i\n"
        "current_piece->y: %i\n"
        "is_last_rotate: %i\n",
        tetris->state->drop_timer,
        tetris->state->das_timer,
        tetris->state->arr_timer,
        tetris->state->soft_drop_timer,
        tetris->state->lock_timer,
        tetris->state->lock_times_left,
        tetris->state->is_left_pressed,
        tetris->state->is_right_pressed,
        tetris->state->is_soft_drop_pressed,
        tetris->state->is_grounded,
        tetris->state->is_update_clear_rows_needed,
        tetris->state->attack_type,
        tetris->state->is_pc,
        tetris->game->current_piece->x,
        tetris->game->current_piece->y,
        tetris->game->state->is_last_rotate
    );

    DrawText(
        buffer, 
        10, 
        10,
        15, 
        DARKGRAY
    );
}

void draw_attack(Tetris* tetris) {
    Game* game = tetris->game;

    if (game->state->hold_piece == NULL) return;
    
    int blockSize = tetris->config->block_size;

    int boardOffsetX = (tetris->config->width - game->board->width * blockSize) / 2;
    int boardOffsetY = (tetris->config->height - game->board->height * blockSize) / 2;    

    int atkOffsetX = boardOffsetX - (game->board->width + 2) * blockSize / 2;
    int atkOffsetY = boardOffsetY + 7 * blockSize;

    char buffer[64];

    char ren_str[16];
    if (tetris->game->state->ren >= 1) {
        snprintf(ren_str, 16, "REN: %i", tetris->game->state->ren);
    } else {
        ren_str[0] = '\0';
    }

    char b2b_str[16];
    if (tetris->state->b2b_count >= 1) {
        snprintf(b2b_str, 16, "B2B: %i", tetris->state->b2b_count);
    } else {
        b2b_str[0] = '\0';
    }


    char atk_str[16];
    if (tetris->state->atk_count >= 1) {
        snprintf(atk_str, 16, "ATK: %i", tetris->state->atk_count);
    } else {
        atk_str[0] = '\0';
    }

    snprintf(buffer, 64,
        "%s\n%s\n%s\n%s\n%s",
        TETRIS_ATK_STR[tetris->state->attack_type],
        tetris->state->is_pc ? "Perfect Clear" : "",
        ren_str,
        b2b_str,
        atk_str
    );

    DrawText(
        buffer, 
        atkOffsetX, 
        atkOffsetY,
        15,     
        DARKGRAY
    );
}

void draw_content(Tetris* tetris) {
    BeginDrawing();
    ClearBackground(RAYWHITE);

    draw_board(tetris);
    if (tetris->config->is_shadow_enabled) draw_shadow(tetris);
    draw_piece(tetris);
    draw_previews(tetris);
    draw_hold_piece(tetris);
    draw_attack(tetris);

    draw_debug_info(tetris);

    if (tetris->state->is_game_over) draw_game_over(tetris);
    EndDrawing();
}

void update_clear_rows(Tetris* tetris) {
    tetris->state->drop_timer = 0.0f;
    tetris->state->is_update_clear_rows_needed = FALSE;

    Game* game = tetris->game;
    printf("update_clear_rows_needed\n");
    
    tetris->state->attack_type = get_attack_type(game);
    tetris->state->is_pc = is_perfect_clear(game);
    update_ren(game);
    update_atk(tetris);
    tetris->state->is_game_over = next_piece(game);
    if (tetris->state->is_game_over) draw_game_over(tetris); // ??
    int clear_count = clear_rows(game->board);
    if (!clear_count > 0) return;
    if (
        tetris->state->attack_type == ATK_SINGLE 
        || tetris->state->attack_type == ATK_DOUBLE
        || tetris->state->attack_type == ATK_TRIPLE
    ) tetris->state->b2b_count = -1;
    else tetris->state->b2b_count++;
}

void run_game(Game* game) {
    Tetris* tetris = init_tetris(game);

    init_window(tetris);
    SetTargetFPS(tetris->config->fps);
    while (!WindowShouldClose()) {
        detect_input(tetris);
        if (tetris->state->is_game_over) draw_game_over(tetris); // ??
        
        update_drop_timer(tetris);
        if (tetris->state->is_update_clear_rows_needed) update_clear_rows(tetris);
        draw_content(tetris);
    }
}

int main() {
    Game* game = init_game(TRUE);
    run_game(game);
    free_game(game);
    CloseWindow();
    return 0;
}