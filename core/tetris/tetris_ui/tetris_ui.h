// core/tetris/tetris_ui/tetris_ui.h

#ifndef TETRIS_UI_H
#define TETRIS_UI_H

#include "../tetris.h"

typedef struct {
    int width;
    int height;
    int block_size;
    Bool is_shadow_enabled;
} UIConfig;

UIConfig* init_ui_config();
void free_ui_config(UIConfig* ui_config);
void init_window(Tetris* tetris, UIConfig* ui_config);
void draw_game_over(UIConfig* ui_config);
void draw_content(Tetris* tetris, UIConfig* ui_config);

#endif