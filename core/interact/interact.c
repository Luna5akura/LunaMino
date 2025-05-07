// core/interact/interact.c

#include <raylib.h>
#include <stdlib.h>
#include <stdio.h>
#include "interact.h"

#define TETRIS_WIDTH 800
#define TETRIS_HEIGHT 600
#define BLOCK_SIZE 20

Color GetColorForPieceType(int type) {
    switch (type) {
        case 0: return GRAY;        //
        case 1: return SKYBLUE;     // I
        case 2: return YELLOW;        // O
        case 3: return PURPLE;      // T
        case 4: return GREEN;      // S 
        case 5: return RED;       // Z
        case 6: return BLUE;         // J
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
    config->block_size = BLOCK_SIZE;
    printf("1gravity: %f, 1fps: %d\n", game->ui_config->gravity, game->ui_config->fps);
    config->drop_interval = 1.0f / (game->ui_config->gravity * (float)game->ui_config->fps);   
    printf("1Drop interval: %f\n", config->drop_interval);   
    return config;
}

TetrisState* init_tetris_state(Game* game) {
    TetrisState* state = (TetrisState*)malloc(sizeof(TetrisState));
    state->drop_timer = 0.0f;
    state->is_game_over = FALSE;
    return state;
}

Tetris* init_tetris(Game* game) {
    Tetris* tetris = (Tetris*)malloc(sizeof(Tetris));
    tetris->config = init_tetris_config(game);
    tetris->state = init_tetris_state(game);
    tetris->game = game;
    return tetris;
}

void update_drop_timer(Tetris* tetris, Game* game) {
    tetris->state->drop_timer += GetFrameTime();
    if (tetris->state->drop_timer < tetris->config->drop_interval) return;
    if (!try_move_piece(game, MOVE_DOWN)) tetris->state->is_game_over = next_piece(game);

    // TODO: time-up lock
    tetris->state->drop_timer = 0.0f;
}

void detect_input(Tetris* tetris) {
    Game* game = tetris->game;
    if (IsKeyPressed(KEY_LEFT)) try_move_piece(game, MOVE_LEFT);
    if (IsKeyPressed(KEY_RIGHT)) try_move_piece(game, MOVE_RIGHT);
    if (IsKeyPressed(KEY_DOWN)) try_move_piece(game, MOVE_DOWN);
    if (IsKeyPressed(KEY_Z)) try_rotate_piece(game, ROTATE_CCW);
    if (IsKeyPressed(KEY_X)) try_rotate_piece(game, ROTATE_CW);
    if (IsKeyPressed(KEY_A)) try_rotate_piece(game, ROTATE_180);
    if (IsKeyPressed(KEY_SPACE)) {
        try_move_piece(game, MOVE_HARD_DROP);
        tetris -> state -> is_game_over = next_piece(game);
    }
}

void draw_board(Tetris* tetris,Game* game) {
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

void draw_piece(Tetris* tetris, Game* game) {
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

void draw_previews(Tetris* tetris, Game* game) {
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

void run_game(Game* game) {
    Tetris* tetris = init_tetris(game);

    init_window(tetris);
    SetTargetFPS(game->ui_config->fps);

    while (!WindowShouldClose()) {
        
        detect_input(tetris);

        if (tetris->state->is_game_over == TRUE) exit_window();

        update_drop_timer(tetris, game);
        clear_rows(game->board);

        BeginDrawing();
        ClearBackground(RAYWHITE);

        draw_board(tetris, game);
        draw_piece(tetris, game);
        draw_previews(tetris, game);

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