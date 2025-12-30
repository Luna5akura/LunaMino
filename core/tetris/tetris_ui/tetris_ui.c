#include "tetris_ui.h"
#include <stdlib.h>
#include <stdio.h> // for snprintf if needed

// 攻击类型文本表
const char* TETRIS_ATK_STR[28] = {
    "None", "Single", "Double", "Triple", "Tetris",
    "T-Spin Mini Single", "T-Spin Single", "T-Spin Mini Double", "T-Spin Double", "T-Spin Triple",
    "I-Spin Single", "I-Spin Double", "I-Spin Triple",
    "O-Spin Single", "O-Spin Double",
    "S-Spin Single", "S-Spin Double", "S-Spin Triple",
    "Z-Spin Single", "Z-Spin Double", "Z-Spin Triple",
    "L-Spin Single", "L-Spin Double", "L-Spin Triple",
    "J-Spin Single", "J-Spin Double", "J-Spin Triple",
    "ERROR",
};

// 方块颜色表 (R, G, B, A)
static const Color PIECE_COLORS[] = {
    {0, 240, 240, 255},   // I (Cyan)
    {240, 240, 0, 255},   // O (Yellow)
    {160, 0, 240, 255},   // T (Purple)
    {0, 240, 0, 255},     // S (Green)
    {240, 0, 0, 255},     // Z (Red)
    {0, 0, 240, 255},     // J (Blue)
    {240, 160, 0, 255},   // L (Orange)
    {100, 100, 100, 255}  // Garbage/Ghost (Gray) - index 7 (PIECE_COUNT)
};

Color get_color_from_piece_type(uint8_t type) {
    if (type >= 7) return PIECE_COLORS[7];
    return PIECE_COLORS[type];
}

UIConfig* init_ui_config() {
    UIConfig* c = (UIConfig*)malloc(sizeof(UIConfig));
    if (c) {
        c->width = UI_WIDTH;
        c->height = UI_HEIGHT;
        c->block_size = UI_BLOCK_SIZE;
        c->is_shadow_enabled = UI_IS_SHADOW_ENABLED;
        // 居中计算
        c->board_offset_x = (c->width - BOARD_WIDTH * c->block_size) / 2;
        // 垂直方向通常偏上一点，或者居中
        c->board_offset_y = (c->height - UI_VISIBLE_ROWS * c->block_size) / 2;
    }
    return c;
}

void free_ui_config(UIConfig* ui_config) {
    if (ui_config) free(ui_config);
}

void init_window(const UIConfig* ui_config) {
    InitWindow(ui_config->width, ui_config->height, "Tetris AI Core Optimized");
    SetTargetFPS(60);
}

// 内部绘图辅助：绘制单个方格
// y_inverted: 游戏逻辑的y(0在底部) -> 屏幕y
static inline void draw_block_at(int x, int y, const UIConfig* cfg, Color color, bool outline) {
    // 游戏逻辑中 y=0 是最底部。
    // 屏幕绘制中，base_y 对应顶部。
    // 实际绘制位置 = base_y + (总显示行数 - 1 - 逻辑y) * block_size
    // 注意：BOARD_HEIGHT 是 23，但 UI 只显示 20 行。我们需要偏移。
    
    // 如果 y >= UI_VISIBLE_ROWS，说明在缓冲区，不绘制（除非你想做 debug 显示）
    if (y >= UI_VISIBLE_ROWS) return;

    int draw_x = cfg->board_offset_x + x * cfg->block_size;
    int draw_y = cfg->board_offset_y + (UI_VISIBLE_ROWS - 1 - y) * cfg->block_size;

    DrawRectangle(draw_x, draw_y, cfg->block_size, cfg->block_size, color);
    
    // 轮廓线稍微深一点，增强视觉
    if (outline) {
        Color outlineColor = {0, 0, 0, 40}; // 半透明黑
        DrawRectangleLines(draw_x, draw_y, cfg->block_size, cfg->block_size, outlineColor);
    }
}

// 绘制边框
static void draw_border(const UIConfig* cfg) {
    DrawRectangleLines(
        cfg->board_offset_x - 2,
        cfg->board_offset_y - 2,
        BOARD_WIDTH * cfg->block_size + 4,
        UI_VISIBLE_ROWS * cfg->block_size + 4,
        LIGHTGRAY
    );
}

void draw_board(const Game* game, const UIConfig* cfg) {
    draw_border(cfg);

    Color stackColor = {200, 200, 200, 255}; // 已锁定方块的颜色 (也可以根据逻辑存颜色，这里简化为统一灰)

    // 优化：外层循环计算 Y 坐标，内层只算 X
    // 只遍历可见区域
    for (int y = 0; y < UI_VISIBLE_ROWS; y++) {
        // 快速跳过空行 (row == 0)
        if (game->board.rows[y] == 0) continue;

        for (int x = 0; x < BOARD_WIDTH; x++) {
            // 位运算检查
            if ((game->board.rows[y] >> x) & 1) {
                // 如果你有彩色棋盘的需求，这里需要从单独的 board_colors 数组读取
                // 这里暂时用统一颜色
                draw_block_at(x, y, cfg, stackColor, true);
            }
        }
    }
}

// 通用方块绘制
// is_grid_coords: true 表示 p->pos 是棋盘坐标；false 表示 p->pos 无效，直接画在 origin_x/y (用于预览/Hold)
static void draw_piece_generic(const Piece* p, int origin_x, int origin_y, const UIConfig* cfg, Color color, bool is_grid_coords) {
    uint16_t mask = piece_get_mask(p);
    
    // 遍历 mask 的 4 行
    for (int r = 0; r < 4; r++) {
        uint16_t row_data = (mask >> ((3 - r) * 4)) & 0xF;
        if (!row_data) continue;
        
        for (int c = 0; c < 4; c++) {
            // 检查 mask 的位 (注意 mask 是从高位开始)
            if ((row_data >> (3 - c)) & 1) {
                if (is_grid_coords) {
                    // --- 核心修复：使用 pos[0/1] ---
                    int bx = p->pos[0] + c;
                    int by = p->pos[1] - r;
                    
                    // 边界检查，防止画出界
                    if (bx >= 0 && bx < BOARD_WIDTH && by >= 0) {
                        draw_block_at(bx, by, cfg, color, true);
                    }
                } else {
                    // 预览框绘制模式
                    // 居中偏移 logic (可选)
                    int draw_x = origin_x + c * cfg->block_size;
                    int draw_y = origin_y + r * cfg->block_size;
                    
                    DrawRectangle(draw_x, draw_y, cfg->block_size, cfg->block_size, color);
                    DrawRectangleLines(draw_x, draw_y, cfg->block_size, cfg->block_size, (Color){0,0,0,50});
                }
            }
        }
    }
}

void draw_piece(const Game* game, const UIConfig* cfg) {
    Color color = get_color_from_piece_type(game->current_piece.type);
    draw_piece_generic(&game->current_piece, 0, 0, cfg, color, true);
}

void draw_shadow(const Game* game, const UIConfig* cfg) {
    int shadow_height = game_get_shadow_height(game);
    // 如果已经在底部，不需要画阴影
    // 或者 shadow_height 很大说明逻辑有问题
    if (shadow_height <= 0) return;

    Piece shadow_piece = game->current_piece;
    // --- 核心修复：修改 pos[1] ---
    shadow_piece.pos[1] -= shadow_height;

    // 获取颜色并增加透明度
    Color color = get_color_from_piece_type(shadow_piece.type);
    color.a = 60; // 透明

    draw_piece_generic(&shadow_piece, 0, 0, cfg, color, true);
}

void draw_previews(const Game* game, const UIConfig* cfg) {
    int start_x = cfg->board_offset_x + (BOARD_WIDTH + 1) * cfg->block_size;
    int start_y = cfg->board_offset_y;

    DrawText("NEXT", start_x, start_y - 20, 20, DARKGRAY);

    for (int i = 0; i < game->config.preview_count; i++) {
        // 使用 peek 查看
        PieceType type = previews_peek(&game->state.previews, i);
        
        Piece temp;
        piece_init(&temp, type);
        // piece_init 默认会给 spawn position，这里不需要，我们只用 mask
        // 但需要注意：预览时通常旋转为 0

        Color color = get_color_from_piece_type(type);
        
        // 简单垂直排列，每个占 3 格高 (虽然方块是 4 格，但预览紧凑点好看)
        int y_pos = start_y + i * 3 * cfg->block_size + 10;
        
        draw_piece_generic(&temp, start_x, y_pos, cfg, color, false);
    }
}

void draw_hold_piece(const Game* game, const UIConfig* cfg) {
    int start_x = cfg->board_offset_x - 5 * cfg->block_size;
    int start_y = cfg->board_offset_y;

    DrawText("HOLD", start_x, start_y - 20, 20, DARKGRAY);

    if (game->state.has_hold_piece) {
        Piece temp = game->state.hold_piece;
        // Hold 的方块通常重置为默认旋转
        temp.rotation = 0; 

        Color color = get_color_from_piece_type(temp.type);
        if (!game->state.can_hold_piece) {
            color = GRAY; // 如果不能交换，变灰
        }
        
        draw_piece_generic(&temp, start_x, start_y + 10, cfg, color, false);
    }
}

void draw_attack(const Tetris* tetris, const UIConfig* cfg) {
    int x = cfg->board_offset_x - 6 * cfg->block_size;
    int y = cfg->board_offset_y + 8 * cfg->block_size; // 显示在 Hold 下方

    DrawText("STATS", x, y, 20, DARKGRAY); y += 30;
    
    // 显示 B2B
    if (tetris->state.b2b_count > 0) {
        DrawText(TextFormat("B2B: %d", tetris->state.b2b_count), x, y, 20, ORANGE);
    }
    y += 25;

    // 显示 Combo (REN) - 注意 Game 结构体里是 ren
    if (tetris->game.state.ren > 0) {
        DrawText(TextFormat("Combo: %d", tetris->game.state.ren), x, y, 20, BLUE);
    }
    y += 25;

    // 显示总攻击数
    DrawText(TextFormat("Atk: %d", tetris->state.atk_count), x, y, 20, BLACK); y += 25;
    
    // 显示垃圾行缓存
    if (tetris->state.pending_attack > 0) {
        DrawText(TextFormat("Pending: %d", tetris->state.pending_attack), x, y, 20, RED);
    }
    y += 35;

    // 显示最后一次攻击类型
    if (tetris->state.attack_type != ATK_NONE) {
        // 防止数组越界
        if (tetris->state.attack_type >= 0 && tetris->state.attack_type < 28) {
            DrawText(TETRIS_ATK_STR[tetris->state.attack_type], x, y, 20, PURPLE);
        }
    }
    
    // PC 判定
    if (tetris->state.is_pc) {
        DrawText("PERFECT CLEAR!", x, y + 25, 20, GOLD);
    }
}

void draw_debug_info(const Tetris* tetris) {
    DrawFPS(10, 10);
    // 假设 Tetris 结构体里有 lock_timer 和 config
    DrawText(TextFormat("Lock: %.2f", tetris->state.lock_timer), 10, 30, 10, GRAY);
    
    // --- 核心修复：使用 pos[0/1] ---
    DrawText(TextFormat("Pos: (%d, %d)", 
        tetris->game.current_piece.pos[0], 
        tetris->game.current_piece.pos[1]), 
        10, 45, 10, GRAY);
}

void draw_game_over(const UIConfig* cfg) {
    // 半透明遮罩
    DrawRectangle(0, 0, cfg->width, cfg->height, (Color){0, 0, 0, 150});

    const char* text = "GAME OVER";
    int fontSize = 40;
    int textWidth = MeasureText(text, fontSize);
    DrawText(text, (cfg->width - textWidth) / 2, cfg->height / 2 - 40, fontSize, RED);
    
    const char* subtext = "Press R to Restart";
    int subWidth = MeasureText(subtext, 20);
    DrawText(subtext, (cfg->width - subWidth) / 2, cfg->height / 2 + 20, 20, WHITE);
}

void draw_content(const Tetris* tetris, const UIConfig* cfg) {
    BeginDrawing();
    ClearBackground(RAYWHITE);

    // 绘制游戏各部分
    draw_board(&tetris->game, cfg);
    
    if (cfg->is_shadow_enabled) {
        draw_shadow(&tetris->game, cfg);
    }
    
    draw_piece(&tetris->game, cfg);
    draw_previews(&tetris->game, cfg);
    draw_hold_piece(&tetris->game, cfg);
    draw_attack(tetris, cfg);
    draw_debug_info(tetris);

    if (tetris->state.is_game_over) {
        draw_game_over(cfg);
    }

    EndDrawing();
}