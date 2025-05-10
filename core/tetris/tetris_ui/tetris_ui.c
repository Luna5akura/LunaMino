// core/tetris/tetris_ui/tetris_ui.c

#include <raylib.h>
#include <stdlib.h>
#include <stdio.h>
#include "tetris_ui.h"

#define UI_GAME_HEIGHT 20

#define UI_WIDTH 800
#define UI_HEIGHT 600
#define UI_BLOCK_SIZE 20
#define UI_IS_SHADOW_ENABLED TRUE


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

Color get_color_from_piece_type(int type) {
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

UIConfig* init_ui_config() {
    UIConfig* ui_config = (UIConfig*) malloc(sizeof(UIConfig));
    ui_config->width = UI_WIDTH;
    ui_config->height = UI_HEIGHT;
    ui_config->block_size = UI_BLOCK_SIZE;
    ui_config->is_shadow_enabled = UI_IS_SHADOW_ENABLED;
    return ui_config;
}

void free_ui_config(UIConfig* ui_config) {
    free(ui_config);
}

void init_window(Tetris* tetris, UIConfig* ui_config) {
    int screenWidth = ui_config->width;
    int screenHeight = ui_config->height;
    InitWindow(screenWidth, screenHeight, "Tetris Game");
}

void exit_window() {}


void draw_board(Game* game, UIConfig* ui_config) {
    int blockSize = ui_config->block_size;
    int boardOffsetX = (ui_config->width - game->board->width * blockSize) / 2;
    int boardOffsetY = (ui_config->height - game->board->height * blockSize) / 2;
    
    for (int y = 0; y < game->board->height; y++) {
        for (int x = 0; x < game->board->width; x++) {
            if (y >= UI_GAME_HEIGHT && game->board->state[x][y] == 0) continue;

            Color color = get_color_from_piece_type(game->board->state[x][y]);
            
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

void draw_piece(Game* game, UIConfig* ui_config) {
    int blockSize = ui_config->block_size;
    int boardOffsetX = (ui_config->width - game->board->width * blockSize) / 2;
    int boardOffsetY = (ui_config->height - game->board->height * blockSize) / 2;    

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (game->current_piece->shape[i][j] == 0) continue;
            int x = game->current_piece->x + j;
            int y = game->current_piece->y - i;
            if (y < 0) continue;

            Color color = get_color_from_piece_type(game->current_piece->type + 1);
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

void draw_previews(Game* game, UIConfig* ui_config) {
    int blockSize = ui_config->block_size;
    int boardOffsetX = (ui_config->width - game->board->width * blockSize) / 2;
    int boardOffsetY = (ui_config->height - game->board->height * blockSize) / 2;    

    int previewOffsetX = boardOffsetX + (game->board->width + 2) * blockSize;
    int previewOffsetY = boardOffsetY;
    for (int p = 0; p < game->config->preview_count; p++) {
        Piece* preview_piece = init_piece(game->state->previews->previews[(p + game->state->previews->current) % 5]);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (preview_piece->shape[i][j] != 0) {
                    Color color = get_color_from_piece_type(preview_piece->type + 1);
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

void draw_hold_piece(Game* game, UIConfig* ui_config) {
    if (game->state->hold_piece == NULL) return;
    
    int blockSize = ui_config->block_size;

    int boardOffsetX = (ui_config->width - game->board->width * blockSize) / 2;
    int boardOffsetY = (ui_config->height - game->board->height * blockSize) / 2;    

    int holdOffsetX = boardOffsetX - (game->board->width + 2) * blockSize / 2;
    int holdOffsetY = boardOffsetY;

    Piece* hold_piece = game->state->hold_piece;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (hold_piece->shape[i][j] != 0) {
                Color color = get_color_from_piece_type(hold_piece->type + 1);
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

void draw_shadow(Game* game, UIConfig* ui_config) {
    int shadow_height = get_shadow_height(game);
    if (shadow_height == 0) return;


    int blockSize = ui_config->block_size;
    int boardOffsetX = (ui_config->width - game->board->width * blockSize) / 2;
    int boardOffsetY = (ui_config->height - game->board->height * blockSize) / 2;    

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (game->current_piece->shape[i][j] == 0) continue;
            int x = game->current_piece->x + j;
            int y = game->current_piece->y - i;
            if (y < 0) continue;

            Color color = get_color_from_piece_type(game->current_piece->type + 1);
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
        "is_last_rotate: %i\n"
        "is_game_over: %i\n",
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
        tetris->game->state->is_last_rotate,
        tetris->state->is_game_over
    );

    DrawText(
        buffer, 
        10, 
        10,
        15, 
        DARKGRAY
    );
}

void draw_attack(Tetris* tetris, UIConfig* ui_config) {
    Game* game = tetris->game;
    if (game->state->hold_piece == NULL) return;
    
    int blockSize = ui_config->block_size;

    int boardOffsetX = (ui_config->width - game->board->width * blockSize) / 2;
    int boardOffsetY = (ui_config->height - game->board->height * blockSize) / 2;    

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

void draw_game_over(UIConfig* ui_config) {
    DrawText(
        "Game Over", 
        ui_config->width / 2 - MeasureText("Game Over", 40) / 2, 
        ui_config->height / 2 - 20, 
        40, 
        RED
    );
}


void draw_content(Tetris* tetris, UIConfig* ui_config) {
    Game* game = tetris->game;
    BeginDrawing();
    ClearBackground(RAYWHITE);

    draw_board(game, ui_config);
    if (ui_config->is_shadow_enabled) draw_shadow(game, ui_config);
    draw_piece(game, ui_config);
    draw_previews(game, ui_config);
    draw_hold_piece(game, ui_config);
    draw_attack(tetris, ui_config);

    draw_debug_info(tetris);

    if (tetris->state->is_game_over) draw_game_over(ui_config);
    EndDrawing();
}

