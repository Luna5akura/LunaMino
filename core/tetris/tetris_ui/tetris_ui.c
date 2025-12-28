// core/tetris/tetris_ui/tetris_ui.c
#include <raylib.h>
#include <stdlib.h>
#include <stdio.h>
#include "tetris_ui.h"
#include "../../game/game.h" // For game_get_shadow_height, piece_get_mask
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
        case 0: return (Color){0, 0, 0, 0}; // Transparent empty
        case 1: return SKYBLUE; // I
        case 2: return YELLOW; // O
        case 3: return PURPLE; // T
        case 4: return GREEN; // S
        case 5: return RED; // Z
        case 6: return BLUE; // J
        case 7: return ORANGE; // L
        case 8: return DARKGRAY; // Garbage
        default: return BLACK;
    }
}
UIConfig* init_ui_config() {
    UIConfig* ui_config = (UIConfig*)malloc(sizeof(UIConfig));
    if (ui_config) {
        ui_config->width = UI_WIDTH;
        ui_config->height = UI_HEIGHT;
        ui_config->block_size = UI_BLOCK_SIZE;
        ui_config->is_shadow_enabled = UI_IS_SHADOW_ENABLED;
    }
    return ui_config;
}
void free_ui_config(UIConfig* ui_config) {
    free(ui_config);
}
void init_window(const Tetris* tetris, const UIConfig* ui_config) {
    int screenWidth = ui_config->width;
    int screenHeight = ui_config->height;
    InitWindow(screenWidth, screenHeight, "Tetris Game");
}
void draw_board(const Game* game, const UIConfig* ui_config) {
    int blockSize = ui_config->block_size;
    int boardOffsetX = (ui_config->width - game->board.width * blockSize) / 2;
    int boardOffsetY = (ui_config->height - game->board.height * blockSize) / 2;

    for (int y = 0; y < game->board.height; y++) {
        for (int x = 0; x < game->board.width; x++) {
            int cell = board_get_cell(&game->board, x, y);
            if (y >= UI_GAME_HEIGHT && cell == 0) continue;
            Color color = get_color_from_piece_type(cell);
            DrawRectangle(
                boardOffsetX + x * blockSize,
                boardOffsetY + (game->board.height - y - 1) * blockSize,
                blockSize,
                blockSize,
                color
            );
            DrawRectangleLines(
                boardOffsetX + x * blockSize,
                boardOffsetY + (game->board.height - y - 1) * blockSize,
                blockSize,
                blockSize,
                DARKGRAY
            );
        }
    }
}
static void draw_piece_mask(const Piece* p, int offsetX, int offsetY, int blockSize, Color color, Bool is_lines) {
    uint16_t mask = piece_get_mask(p);
    for (int dy = 0; dy < 4; dy++) {
        for (int dx = 0; dx < 4; dx++) {
            if (mask & (1u << (15 - (dy * 4 + dx)))) {
                DrawRectangle(
                    offsetX + dx * blockSize,
                    offsetY + dy * blockSize,
                    blockSize,
                    blockSize,
                    color
                );
                if (is_lines) {
                    DrawRectangleLines(
                        offsetX + dx * blockSize,
                        offsetY + dy * blockSize,
                        blockSize,
                        blockSize,
                        DARKGRAY
                    );
                }
            }
        }
    }
}
void draw_piece(const Game* game, const UIConfig* ui_config) {
    int blockSize = ui_config->block_size;
    int boardOffsetX = (ui_config->width - game->board.width * blockSize) / 2;
    int boardOffsetY = (ui_config->height - game->board.height * blockSize) / 2;
    Piece temp = game->current_piece;
    uint16_t mask = piece_get_mask(&temp);
    for (int dy = 0; dy < 4; dy++) {
        for (int dx = 0; dx < 4; dx++) {
            if (mask & (1u << (15 - (dy * 4 + dx)))) {
                int bx = temp.x + dx;
                int by = temp.y - dy;
                if (by < 0) continue;
                Color color = get_color_from_piece_type(temp.type + 1);
                DrawRectangle(
                    boardOffsetX + bx * blockSize,
                    boardOffsetY + (game->board.height - by - 1) * blockSize,
                    blockSize,
                    blockSize,
                    color
                );
                DrawRectangleLines(
                    boardOffsetX + bx * blockSize,
                    boardOffsetY + (game->board.height - by - 1) * blockSize,
                    blockSize,
                    blockSize,
                    DARKGRAY
                );
            }
        }
    }
}
void draw_previews(const Game* game, const UIConfig* ui_config) {
    int blockSize = ui_config->block_size;
    int boardOffsetX = (ui_config->width - game->board.width * blockSize) / 2;
    int boardOffsetY = (ui_config->height - game->board.height * blockSize) / 2;
    int previewOffsetX = boardOffsetX + (game->board.width + 2) * blockSize;
    int previewOffsetY = boardOffsetY;
    for (int p = 0; p < game->config.preview_count; p++) {
        PieceType type = previews_peek(&game->state.previews, p);
        Piece preview;
        piece_init(&preview, type);
        Color color = get_color_from_piece_type(preview.type + 1);
        int pieceOffsetY = previewOffsetY + p * 5 * blockSize; // Space between previews
        draw_piece_mask(&preview, previewOffsetX, pieceOffsetY, blockSize, color, true);
    }
}
void draw_hold_piece(const Game* game, const UIConfig* ui_config) {
    if (!game->state.has_hold_piece) return;
    int blockSize = ui_config->block_size;
    int boardOffsetX = (ui_config->width - game->board.width * blockSize) / 2;
    int boardOffsetY = (ui_config->height - game->board.height * blockSize) / 2;
    int holdOffsetX = boardOffsetX - 6 * blockSize; // Adjust offset
    int holdOffsetY = boardOffsetY;
    Piece temp = game->state.hold_piece;
    temp.rotation = 0; // Reset for display
    Color color = get_color_from_piece_type(temp.type + 1);
    draw_piece_mask(&temp, holdOffsetX, holdOffsetY, blockSize, color, true);
}
void draw_shadow(const Game* game, const UIConfig* ui_config) {
    int shadow_height = game_get_shadow_height(game);
    if (shadow_height <= 0) return;
    int blockSize = ui_config->block_size;
    int boardOffsetX = (ui_config->width - game->board.width * blockSize) / 2;
    int boardOffsetY = (ui_config->height - game->board.height * blockSize) / 2;
    Piece temp = game->current_piece;
    Color color = get_color_from_piece_type(temp.type + 1);
    color.a = 85; // Semi-transparent
    uint16_t mask = piece_get_mask(&temp);
    for (int dy = 0; dy < 4; dy++) {
        for (int dx = 0; dx < 4; dx++) {
            if (mask & (1u << (15 - (dy * 4 + dx)))) {
                int bx = temp.x + dx;
                int by = temp.y - dy - shadow_height;
                if (by < 0) continue;
                DrawRectangle(
                    boardOffsetX + bx * blockSize,
                    boardOffsetY + (game->board.height - by - 1) * blockSize,
                    blockSize,
                    blockSize,
                    color
                );
                DrawRectangleLines(
                    boardOffsetX + bx * blockSize,
                    boardOffsetY + (game->board.height - by - 1) * blockSize,
                    blockSize,
                    blockSize,
                    DARKGRAY
                );
            }
        }
    }
}
void draw_debug_info(const Tetris* tetris) {
    char buffer[2048];
    snprintf(buffer, sizeof(buffer),
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
        "current_piece.x: %i\n"
        "current_piece.y: %i\n"
        "is_last_rotate: %i\n"
        "is_game_over: %i\n"
        "undo_timer: %.2f\n",
        tetris->state.drop_timer,
        tetris->state.das_timer,
        tetris->state.arr_timer,
        tetris->state.soft_drop_timer,
        tetris->state.lock_timer,
        tetris->state.lock_times_left,
        tetris->state.is_left_pressed,
        tetris->state.is_right_pressed,
        tetris->state.is_soft_drop_pressed,
        tetris->state.is_grounded,
        tetris->state.is_update_clear_rows_needed,
        tetris->state.attack_type,
        tetris->state.is_pc,
        tetris->game.current_piece.x,
        tetris->game.current_piece.y,
        tetris->game.state.is_last_rotate,
        tetris->state.is_game_over,
        tetris->state.undo_timer
    );
    DrawText(buffer, 10, 10, 15, DARKGRAY);
}
void draw_attack(const Tetris* tetris, const UIConfig* ui_config) {
    int blockSize = ui_config->block_size;
    int boardOffsetX = (ui_config->width - tetris->game.board.width * blockSize) / 2;
    int boardOffsetY = (ui_config->height - tetris->game.board.height * blockSize) / 2;
    int atkOffsetX = boardOffsetX - 6 * blockSize;
    int atkOffsetY = boardOffsetY + 7 * blockSize;
    char buffer[256];
    char ren_str[16] = {0};
    if (tetris->game.state.ren >= 1) snprintf(ren_str, sizeof(ren_str), "REN: %i", tetris->game.state.ren);
    char b2b_str[16] = {0};
    if (tetris->state.b2b_count >= 1) snprintf(b2b_str, sizeof(b2b_str), "B2B: %i", tetris->state.b2b_count);
    char atk_str[16] = {0};
    if (tetris->state.atk_count >= 1) snprintf(atk_str, sizeof(atk_str), "ATK: %i", tetris->state.atk_count);
    snprintf(buffer, sizeof(buffer),
        "%s\n%s\n%s\n%s\n%s",
        TETRIS_ATK_STR[tetris->state.attack_type],
        tetris->state.is_pc ? "Perfect Clear" : "",
        ren_str,
        b2b_str,
        atk_str
    );
    DrawText(buffer, atkOffsetX, atkOffsetY, 15, DARKGRAY);
}
void draw_game_over(const UIConfig* ui_config) {
    DrawText(
        "Game Over",
        ui_config->width / 2 - MeasureText("Game Over", 40) / 2,
        ui_config->height / 2 - 20,
        40,
        RED
    );
}
void draw_content(const Tetris* tetris, const UIConfig* ui_config) {
    BeginDrawing();
    ClearBackground(RAYWHITE);
    draw_board(&tetris->game, ui_config);
    if (ui_config->is_shadow_enabled) draw_shadow(&tetris->game, ui_config);
    draw_piece(&tetris->game, ui_config);
    draw_previews(&tetris->game, ui_config);
    draw_hold_piece(&tetris->game, ui_config);
    draw_attack(tetris, ui_config);
    draw_debug_info(tetris);
    if (tetris->state.is_game_over) draw_game_over(ui_config);
    EndDrawing();
}