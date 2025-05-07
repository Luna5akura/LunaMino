// core/interact/interact.c

#include <raylib.h>
#include <stdlib.h>
#include <stdio.h>
#include "interact.h"

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
    state->is_game_over = FALSE;

    return state;
}

Tetris* init_tetris(Game* game) {
    Tetris* tetris = (Tetris*)malloc(sizeof(Tetris));

    tetris->config = init_tetris_config(game);
    tetris->state = init_tetris_state(game, tetris->config);
    tetris->game = game;

    return tetris;
}

void reset_lock_times_left(Tetris* tetris) {
    tetris->state->lock_times_left = tetris->config->reset_lock_times_limit;
    tetris->state->lock_timer = 0.0f;
}

void flush_lock_timer(Tetris* tetris) {
    tetris->state->lock_timer = 0.0f;
    tetris->state->lock_times_left -= 1;
    if (tetris->state->lock_times_left == -1) {
        next_piece(tetris->game);
        reset_lock_times_left(tetris);
        tetris->state->is_grounded = FALSE;
    }
}

void update_drop_timer(Tetris* tetris) {
    Game* game = tetris->game;

    if (tetris->state->is_grounded) {
        tetris->state->lock_timer += GetFrameTime();
        if (tetris->state->lock_timer * 1000 >= tetris->config->lock_delay) {
            next_piece(game);
            reset_lock_times_left(tetris);
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

void detect_input(Tetris* tetris) {
    Game* game = tetris->game;

    if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_RIGHT)) {
        if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_RIGHT)) {
            Bool is_successful = FALSE;

            if (IsKeyPressed(KEY_LEFT)) is_successful = try_move_piece(game, MOVE_LEFT);
            if (IsKeyPressed(KEY_RIGHT)) is_successful = try_move_piece(game, MOVE_RIGHT);

            if (is_successful) {
                if (tetris->state->is_grounded) {
                    flush_lock_timer(tetris);
                } else {
                    tetris->state->lock_timer = 0.0f;
                }
                tetris->state->is_grounded = is_grounded(game);
            }
            
            tetris->state->das_timer = 0;
        }
        else {
            tetris->state->das_timer += GetFrameTime() * 1000;
            if (tetris->state->das_timer >= tetris->config->das) {
                // begin arr
                tetris->state->arr_timer += GetFrameTime() * 1000;
    
                Bool is_looping = TRUE;
                int move_count = -1;

                while (tetris->state->arr_timer >= tetris->config->arr && is_looping) {
                    if (IsKeyDown(KEY_LEFT)) {
                        is_looping = try_move_piece(game, MOVE_LEFT);
                    }
                    if (IsKeyDown(KEY_RIGHT)) { 
                        is_looping = try_move_piece(game, MOVE_RIGHT);
                    }
                    move_count++;
                    tetris->state->arr_timer -= tetris->config->arr;
                }
                if (tetris->state->is_grounded && move_count > 0) {
                    flush_lock_timer(tetris);
                }
                tetris->state->is_grounded = is_grounded(game);
            }
        }
    }
    else{
        tetris->state->das_timer = 0.0f;
    }

    if (IsKeyDown(KEY_DOWN)) {
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

    if (IsKeyPressed(KEY_Z)) {
        Bool is_successful = try_rotate_piece(game, ROTATE_CCW);
        if (is_successful) {
            if (tetris->state->is_grounded) {
                flush_lock_timer(tetris);
            }
            else {
                tetris->state->lock_timer = 0.0f;
            }
            tetris->state->is_grounded = is_grounded(game);
        }
    }
    if (IsKeyPressed(KEY_X)) {
        Bool is_successful = try_rotate_piece(game, ROTATE_CW);
        if (is_successful) {
            if (tetris->state->is_grounded) {
                flush_lock_timer(tetris);
            }
            else {
                tetris->state->lock_timer = 0.0f;
            }
            tetris->state->is_grounded = is_grounded(game);
        }
    }
    if (IsKeyPressed(KEY_A)) try_rotate_piece(game, ROTATE_180);
    if (IsKeyPressed(KEY_C)) {
        tetris->state->is_game_over = try_hold_piece(game);
    };
    if (IsKeyPressed(KEY_SPACE)) {
        try_move_piece(game, MOVE_HARD_DROP);
        tetris->state->is_game_over = next_piece(game);
        tetris->state->is_grounded = FALSE;
        reset_lock_times_left(tetris);
    }
}

void draw_board(Tetris* tetris) {
    Game* game = tetris->game;

    int blockSize = tetris->config->block_size;
    int boardOffsetX = (tetris->config->width - game->board->width * blockSize) / 2;
    int boardOffsetY = (tetris->config->height - game->board->height * blockSize) / 2;
    
    for (int y = 0; y < game->board->height; y++) {
        for (int x = 0; x < game->board->width; x++) {
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
        free(preview_piece);
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
    char buffer[256];

    snprintf(buffer, 256,
        "drop_timer: %.2f\nlock_timer: %.2f\nlock_times_left: %d\nis_grounded: %d",
        tetris->state->drop_timer,
        tetris->state->lock_timer,
        tetris->state->lock_times_left,
        tetris->state->is_grounded
    );

    DrawText(
        buffer, 
        10, 
        10,
        15, 
        DARKGRAY
    );
}

void run_game(Game* game) {
    Tetris* tetris = init_tetris(game);

    init_window(tetris);
    SetTargetFPS(tetris->config->fps);

    while (!WindowShouldClose()) {
        
        detect_input(tetris);

        if (tetris->state->is_game_over == TRUE) exit_window();

        update_drop_timer(tetris);
        clear_rows(game->board);

        BeginDrawing();
        ClearBackground(RAYWHITE);

        draw_board(tetris);
        if (tetris->config->is_shadow_enabled) draw_shadow(tetris);
        draw_piece(tetris);
        draw_previews(tetris);
        draw_hold_piece(tetris);
        draw_debug_info(tetris);

        if (tetris->state->is_game_over == TRUE) {
            DrawText(
                "Game Over", 
                tetris->config->width / 2 - MeasureText("Game Over", 40) / 2, 
                tetris->config->height / 2 - 20, 
                40, 
                RED
            );
        }

        EndDrawing();
    }
}

int main() {
    Game* game = init_game(TRUE);
    run_game(game);
    
    free(game->current_piece);
    free(game->state->previews);
    free(game->state->bag);
    free(game->state);
    free(game->config);
    free(game->board);
    free(game);
    CloseWindow();

    return 0;
}