// core/tetris/tetris_ui/tetris_ui.h

#ifndef TETRIS_UI_H
#define TETRIS_UI_H

#include <raylib.h>
#include "../tetris.h" // 假设这里包含 Tetris 结构体定义 (包含 Game game, TetrisState state 等)

// UI Layout Constants
#define UI_WIDTH 800
#define UI_HEIGHT 600
#define UI_BLOCK_SIZE 28 // 稍微调大以适应视口
#define UI_IS_SHADOW_ENABLED true
#define UI_VISIBLE_ROWS 20 // 游戏实际可见高度

typedef struct {
    int width;
    int height;
    int block_size;
    bool is_shadow_enabled;
    // 预计算的偏移量，避免每帧重复计算
    int board_offset_x;
    int board_offset_y;
} UIConfig;

// Expose the Attack String table
extern const char* TETRIS_ATK_STR[28];

// Lifecycle
UIConfig* init_ui_config();
void free_ui_config(UIConfig* ui_config);
void init_window(const UIConfig* ui_config); // 不需要传 Tetris

// Draw Functions
void draw_content(const Tetris* tetris, const UIConfig* ui_config);

// Helper
Color get_color_from_piece_type(uint8_t type);

#endif