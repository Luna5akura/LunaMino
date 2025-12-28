// core/tetris/tetris_ui/tetris_ui.h

#ifndef TETRIS_UI_H
#define TETRIS_UI_H
#include <raylib.h> // Needed for Color
#include "../tetris.h"
// Move constants here so battle.c can see them
#define UI_WIDTH 800
#define UI_HEIGHT 600
#define UI_BLOCK_SIZE 20
#define UI_IS_SHADOW_ENABLED TRUE
#define UI_GAME_HEIGHT 20
typedef struct {
    int width;
    int height;
    int block_size;
    Bool is_shadow_enabled;
} UIConfig;
// Expose the Attack String table
extern const char TETRIS_ATK_STR[28][20];
UIConfig* init_ui_config();
void free_ui_config(UIConfig* ui_config);
void init_window(const Tetris* tetris, const UIConfig* ui_config);
void draw_game_over(const UIConfig* ui_config);
void draw_content(const Tetris* tetris, const UIConfig* ui_config);
// Expose the color helper
Color get_color_from_piece_type(int type);
#endif