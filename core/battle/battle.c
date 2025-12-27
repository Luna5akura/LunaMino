// core/battle/battle.c

#include "battle.h"
#include <raylib.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../tetris/tetris_ui/tetris_ui.h"
#include "../tetris/tetris_history/tetris_history.h"

// ==========================================
// 1. 初始化与配置函数
// ==========================================

BattleConfig* init_battle_config() {
    BattleConfig* config = (BattleConfig*)malloc(sizeof(BattleConfig));
    config->player_count = 2;
    config->seed1 = 0;
    config->seed2 = 0;
    return config;
}

void free_battle_config(BattleConfig* config) {
    free(config);
}

BattleState* init_battle_state() {
    BattleState* state = (BattleState*)malloc(sizeof(BattleState));
    state->score_player1 = 0;
    state->score_player2 = 0;
    state->is_game_over = FALSE;
    return state;
}

void free_battle_state(BattleState* state) {
    free(state);
}

Battle* init_battle(BattleConfig* config) {
    Battle* battle = (Battle*)malloc(sizeof(Battle));
    battle->config = config;
    battle->state = init_battle_state();

    Game* game1 = init_game();
    battle->player1 = init_tetris(game1);

    Game* game2 = init_game();
    battle->player2 = init_tetris(game2);

    return battle;
}

void free_battle(Battle* battle) {
    free_tetris(battle->player1);
    free_tetris(battle->player2);
    free_battle_state(battle->state);
    free_battle_config(battle->config);
    free(battle);
}

UIConfig* init_battle_ui_config() {
    UIConfig* ui_config = (UIConfig*)malloc(sizeof(UIConfig));
    ui_config->width = 1200;
    ui_config->height = 600;
    ui_config->block_size = UI_BLOCK_SIZE;
    ui_config->is_shadow_enabled = UI_IS_SHADOW_ENABLED;
    return ui_config;
}

void init_battle_window(Battle* battle, UIConfig* ui_config) {
    InitWindow(ui_config->width, ui_config->height, "Tetris Battle");
}

// ==========================================
// 2. 渲染辅助函数
// ==========================================

int get_board_offset_x(UIConfig* ui_config, Game* game, int offset_x_base) {
    int blockSize = ui_config->block_size;
    return offset_x_base + (ui_config->width / 2 - game->board->width * blockSize) / 2;
}

void draw_board_with_offset(Game* game, UIConfig* ui_config, int offset_x_base) {
    int blockSize = ui_config->block_size;
    int boardOffsetX = get_board_offset_x(ui_config, game, offset_x_base);
    int boardOffsetY = (ui_config->height - game->board->height * blockSize) / 2;
    
    for (int y = 0; y < game->board->height; y++) {
        for (int x = 0; x < game->board->width; x++) {
            if (y >= UI_GAME_HEIGHT && game->board->state[x][y] == 0) continue;
            Color color = get_color_from_piece_type(game->board->state[x][y]);
            DrawRectangle(boardOffsetX + x * blockSize, boardOffsetY + (game->board->height - y - 1) * blockSize, blockSize, blockSize, color);
            DrawRectangleLines(boardOffsetX + x * blockSize, boardOffsetY + (game->board->height - y - 1) * blockSize, blockSize, blockSize, DARKGRAY);
        }
    }
}

void draw_piece_with_offset(Game* game, UIConfig* ui_config, int offset_x_base) {
    int blockSize = ui_config->block_size;
    int boardOffsetX = get_board_offset_x(ui_config, game, offset_x_base);
    int boardOffsetY = (ui_config->height - game->board->height * blockSize) / 2;    

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (game->current_piece->shape[i][j] == 0) continue;
            int x = game->current_piece->x + j;
            int y = game->current_piece->y - i;
            if (y < 0) continue;
            Color color = get_color_from_piece_type(game->current_piece->type + 1);
            DrawRectangle(boardOffsetX + x * blockSize, boardOffsetY + (game->board->height - y - 1) * blockSize, blockSize, blockSize, color);
            DrawRectangleLines(boardOffsetX + x * blockSize, boardOffsetY + (game->board->height - y - 1) * blockSize, blockSize, blockSize, DARKGRAY);
        }
    }
}

void draw_previews_with_offset(Game* game, UIConfig* ui_config, int offset_x_base) {
    int blockSize = ui_config->block_size;
    int boardOffsetX = get_board_offset_x(ui_config, game, offset_x_base);
    int boardOffsetY = (ui_config->height - game->board->height * blockSize) / 2;    
    int previewOffsetX = boardOffsetX + (game->board->width + 2) * blockSize;
    int previewOffsetY = boardOffsetY;
    
    for (int p = 0; p < game->config->preview_count; p++) {
        Piece* preview_piece = init_piece(game->state->previews->previews[(p + game->state->previews->current) % 5]);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (preview_piece->shape[i][j] != 0) {
                    Color color = get_color_from_piece_type(preview_piece->type + 1);
                    DrawRectangle(previewOffsetX + j * blockSize, previewOffsetY + p * 4 * blockSize + i * blockSize, blockSize, blockSize, color);
                    DrawRectangleLines(previewOffsetX + j * blockSize, previewOffsetY + p * 4 * blockSize + i * blockSize, blockSize, blockSize, DARKGRAY);
                }
            }
        }
        free_piece(preview_piece);
    }
}

void draw_hold_piece_with_offset(Game* game, UIConfig* ui_config, int offset_x_base) {
    if (game->state->hold_piece == NULL) return;
    int blockSize = ui_config->block_size;
    int boardOffsetX = get_board_offset_x(ui_config, game, offset_x_base);
    int boardOffsetY = (ui_config->height - game->board->height * blockSize) / 2;    
    int holdOffsetX = boardOffsetX - (game->board->width + 2) * blockSize / 2 - 2 * blockSize;
    int holdOffsetY = boardOffsetY;

    Piece* hold_piece = game->state->hold_piece;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (hold_piece->shape[i][j] != 0) {
                Color color = get_color_from_piece_type(hold_piece->type + 1);
                DrawRectangle(holdOffsetX + j * blockSize, holdOffsetY + i * blockSize, blockSize, blockSize, color);
                DrawRectangleLines(holdOffsetX + j * blockSize, holdOffsetY + i * blockSize, blockSize, blockSize, DARKGRAY);
            }
        }
    }
}

void draw_shadow_with_offset(Game* game, UIConfig* ui_config, int offset_x_base) {
    int shadow_height = get_shadow_height(game);
    if (shadow_height == 0) return;
    int blockSize = ui_config->block_size;
    int boardOffsetX = get_board_offset_x(ui_config, game, offset_x_base);
    int boardOffsetY = (ui_config->height - game->board->height * blockSize) / 2;    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (game->current_piece->shape[i][j] == 0) continue;
            int x = game->current_piece->x + j;
            int y = game->current_piece->y - i;
            if (y < 0) continue;
            Color color = get_color_from_piece_type(game->current_piece->type + 1);
            color.a /= 3;
            DrawRectangle(boardOffsetX + x * blockSize, boardOffsetY + (game->board->height - y - 1 + shadow_height) * blockSize, blockSize, blockSize, color);
            DrawRectangleLines(boardOffsetX + x * blockSize, boardOffsetY + (game->board->height - y - 1 + shadow_height) * blockSize, blockSize, blockSize, DARKGRAY);
        }
    }
}

void draw_attack_with_offset(Tetris* tetris, UIConfig* ui_config, int offset_x_base) {
    Game* game = tetris->game;
    int blockSize = ui_config->block_size;
    int boardOffsetX = get_board_offset_x(ui_config, game, offset_x_base);
    int boardOffsetY = (ui_config->height - game->board->height * blockSize) / 2;    
    int atkOffsetX = boardOffsetX - (game->board->width + 2) * blockSize / 2;
    int atkOffsetY = boardOffsetY + 7 * blockSize;

    char buffer[128];
    char ren_str[16] = "";
    if (tetris->game->state->ren >= 1) snprintf(ren_str, 16, "REN: %i", tetris->game->state->ren);

    char b2b_str[16] = "";
    if (tetris->state->b2b_count >= 1) snprintf(b2b_str, 16, "B2B: %i", tetris->state->b2b_count);

    char atk_str[16] = "";
    if (tetris->state->atk_count >= 1) snprintf(atk_str, 16, "ATK: %i", tetris->state->atk_count);
    
    char pending_str[16] = "";
    if (tetris->state->pending_attack > 0) snprintf(pending_str, 16, "Garbage: %i", tetris->state->pending_attack);

    snprintf(buffer, 128, "%s\n%s\n%s\n%s\n%s\n%s", 
        TETRIS_ATK_STR[tetris->state->attack_type], 
        tetris->state->is_pc ? "Perfect Clear" : "", 
        ren_str, b2b_str, atk_str, pending_str
    );

    DrawText(buffer, atkOffsetX, atkOffsetY, 15, DARKGRAY);
}

void draw_debug_info_with_offset(Tetris* tetris, int offset_x) {
    char buffer[128];
    snprintf(buffer, 128, "Lock: %.2f", tetris->state->lock_timer);
    DrawText(buffer, offset_x + 10, 10, 15, DARKGRAY);
}

void draw_content_battle(Battle* battle, UIConfig* ui_config) {
    BeginDrawing();
    ClearBackground(RAYWHITE);

    int offset_p1 = 0;
    int offset_p2 = ui_config->width / 2;

    // P1
    draw_board_with_offset(battle->player1->game, ui_config, offset_p1);
    if (ui_config->is_shadow_enabled) draw_shadow_with_offset(battle->player1->game, ui_config, offset_p1);
    draw_piece_with_offset(battle->player1->game, ui_config, offset_p1);
    draw_previews_with_offset(battle->player1->game, ui_config, offset_p1);
    draw_hold_piece_with_offset(battle->player1->game, ui_config, offset_p1);
    draw_attack_with_offset(battle->player1, ui_config, offset_p1);
    draw_debug_info_with_offset(battle->player1, offset_p1);

    // P2
    draw_board_with_offset(battle->player2->game, ui_config, offset_p2);
    if (ui_config->is_shadow_enabled) draw_shadow_with_offset(battle->player2->game, ui_config, offset_p2);
    draw_piece_with_offset(battle->player2->game, ui_config, offset_p2);
    draw_previews_with_offset(battle->player2->game, ui_config, offset_p2);
    draw_hold_piece_with_offset(battle->player2->game, ui_config, offset_p2);
    draw_attack_with_offset(battle->player2, ui_config, offset_p2);
    draw_debug_info_with_offset(battle->player2, offset_p2);

    if (battle->player1->state->is_game_over) DrawText("P1 LOST", offset_p1 + 100, ui_config->height / 2, 40, RED);
    if (battle->player2->state->is_game_over) DrawText("P2 LOST", offset_p2 + 100, ui_config->height / 2, 40, RED);

    EndDrawing();
}

// ==========================================
// 4. 逻辑与运行
// ==========================================
void ai_step(Tetris* tetris, InputControl actions) {
    Game* game = tetris->game;
    
    // 处理旋转
    if (actions.rotate_cw) {
        if (try_rotate_piece(game, ROTATE_CW)) {
            tetris->state->is_grounded = is_grounded(game);
            if (tetris->state->is_grounded) flush_lock_timer(tetris);
        }
    }
    
    // 处理移动
    if (actions.left) {
         if (try_move_piece(game, MOVE_LEFT)) {
             tetris->state->is_grounded = is_grounded(game);
             if (tetris->state->is_grounded) flush_lock_timer(tetris);
         }
    }
    if (actions.right) {
         if (try_move_piece(game, MOVE_RIGHT)) {
             tetris->state->is_grounded = is_grounded(game);
             if (tetris->state->is_grounded) flush_lock_timer(tetris);
         }
    }
    
    // 处理硬降
    if (actions.hard_drop) {
        detect_hard_drop(tetris); // 这个函数内部只调用 try_move 逻辑，没有 Raylib 依赖，可以用
    }
}

void update_clear_rows_battle(Tetris* self, Tetris* opponent) {
    int attack;
    self->state->drop_timer = 0.0f;
    self->state->is_update_clear_rows_needed = FALSE;

    Game* game = self->game;
    
    self->state->attack_type = get_attack_type(game);
    self->state->is_pc = is_perfect_clear(game);
    update_ren(game);
    attack = get_atk(self);
    self->state->atk_count = attack;
    self->state->is_game_over = next_piece(game);
    
    // B2B Logic check for UI purposes mostly, actual logic is in get_atk
    // This is just to update the counter for display if needed
    if (self->state->attack_type == ATK_SINGLE || 
        self->state->attack_type == ATK_DOUBLE || 
        self->state->attack_type == ATK_TRIPLE) {
        self->state->b2b_count = -1; // Reset B2B
    } else if (self->state->attack_type != ATK_NONE) {
        // Spins, Tetris, etc maintain or increase
        // Note: get_atk handles the logic of whether it increases.
        // Here we just increment for display consistency if it wasn't a reset
        self->state->b2b_count++; 
    }

    int clear_count = clear_rows(game->board);

    if (clear_count == 0) {
        if (self->state->pending_attack == 0) return;
        int lines_to_receive = self->state->pending_attack > 8 ? 8 : self->state->pending_attack;
        receive_garbage_line(self, lines_to_receive);
        self->state->pending_attack -= lines_to_receive;
    }
    else {
        if (self->state->pending_attack == 0) {
            receive_attack(opponent, attack);
        }
        else {
            if (attack > self->state->pending_attack) {
                receive_attack(opponent, attack - self->state->pending_attack);
                self->state->pending_attack = 0;
            }
            else {
                self->state->pending_attack -= attack;
            }
        }
    }
}

void run_battle() {
    BattleConfig* config = init_battle_config();
    Battle* battle = init_battle(config);
    UIConfig* ui_config = init_battle_ui_config();

    init_battle_window(battle, ui_config);
    SetTargetFPS(battle->player1->config->fps);

    TetrisHistory* history1 = init_tetris_history(300);

    // 定义 P1 控制，注意 restart 设为 0，由 battle 统一接管
    InputControl controls_p1 = {
        .left = KEY_LEFT, .right = KEY_RIGHT, .down = KEY_DOWN,
        .hard_drop = KEY_SPACE,
        .rotate_cw = KEY_X, .rotate_ccw = KEY_Z, .rotate_180 = KEY_A,
        .hold = KEY_C, .undo = KEY_Q, 
        .restart = 0 // Disable internal restart
    };

    while (!WindowShouldClose()) {
        // 检测全局重启 R
        if (IsKeyPressed(KEY_R)) {
            // 重置 Player 1
            free_tetris(battle->player1);
            Game* g1 = init_game();
            battle->player1 = init_tetris(g1);
            
            // 重置 Player 2
            free_tetris(battle->player2);
            Game* g2 = init_game();
            battle->player2 = init_tetris(g2);

            // 重置 History
            free_tetris_history(history1);
            history1 = init_tetris_history(300);
            
            battle->state->is_game_over = FALSE;
        }

        if (!battle->player1->state->is_game_over) {
            battle->player1 = detect_input(battle->player1, history1, controls_p1);
            update_drop_timer(battle->player1);
            battle->player1->state->undo_timer += GetFrameTime();
        }

        if (battle->player1->state->is_update_clear_rows_needed) {
            update_clear_rows_battle(battle->player1, battle->player2);
            push_history(history1, battle->player1);
        }

        if (battle->player2->state->is_update_clear_rows_needed) {
            update_clear_rows_battle(battle->player2, battle->player1);
        }

        draw_content_battle(battle, ui_config);

        if (battle->player1->state->is_game_over && battle->player2->state->is_game_over) {
            battle->state->is_game_over = TRUE;
        }
    }

    free_tetris_history(history1);
    free_battle(battle);
    free_ui_config(ui_config);
    CloseWindow();
}

int main() {
    run_battle();
    return 0;
}