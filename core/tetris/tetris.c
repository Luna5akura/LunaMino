// core/tetris/tetris.c
#include "tetris.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define DEFAULT_FPS 60
#define DEFAULT_GRAVITY (1.0f / 60.0f)
#define DEFAULT_DAS 150.0f
#define DEFAULT_ARR 30.0f
#define DEFAULT_SOFT_DROP_GRAVITY 0.5f // 稍微调快默认软降速度
#define DEFAULT_UNDO_INTERVAL 150.0f
#define DEFAULT_LOCK_DELAY 500.0f
#define DEFAULT_RESET_LOCK_TIMES_LIMIT 15

// Attack Table: [REN 0-20][AttackType Offset]
// Offset 0=Single, 1=Double, 2=Triple, 3=Tetris, 
// 4=TSMS, 5=TSS, 6=TSMD, 7=TSD, 8=TST
static const uint8_t ATK_TABLE[21][9] = {
    {0, 1, 2, 4, 0, 2, 1, 4, 6}, {0, 1, 2, 5, 0, 2, 1, 5, 7},
    {1, 1, 3, 6, 1, 3, 1, 6, 9}, {1, 1, 3, 7, 1, 3, 1, 7, 10},
    {1, 2, 4, 8, 1, 4, 2, 8, 12}, {1, 2, 4, 9, 1, 4, 2, 9, 13},
    {2, 2, 5, 10, 2, 5, 2, 10, 15}, {2, 2, 5, 11, 2, 5, 2, 11, 16},
    {2, 3, 6, 13, 2, 6, 3, 13, 19}, {2, 3, 7, 14, 2, 7, 3, 14, 21},
    {2, 3, 7, 15, 2, 7, 3, 15, 22}, {2, 4, 8, 16, 2, 8, 4, 16, 24},
    {2, 4, 8, 17, 2, 8, 4, 17, 25}, {2, 4, 9, 18, 2, 9, 4, 18, 27},
    {2, 4, 9, 19, 2, 9, 4, 19, 28}, {3, 5, 10, 20, 3, 10, 5, 20, 30},
    {3, 5, 10, 21, 3, 10, 5, 21, 31}, {3, 5, 11, 22, 3, 11, 5, 22, 33},
    {3, 5, 11, 23, 3, 11, 5, 23, 34}, {3, 6, 12, 24, 3, 12, 6, 24, 36},
    {3, 6, 12, 24, 3, 12, 6, 24, 36} // Max REN cap
};

// B2B Table: [REN 0-20][B2B Type Offset]
// Offset 0=Tetris, 1=TSMS, 2=TSS, 3=TSMD, 4=TSD, 5=TST
static const uint8_t ATK_TABLE_B2B1[21][6] = {
    {5, 1, 3, 2, 5, 7}, {6, 1, 3, 2, 6, 8},
    {7, 1, 4, 3, 7, 10}, {8, 1, 5, 3, 8, 12},
    {10, 2, 6, 4, 10, 14}, {11, 2, 6, 4, 11, 15},
    {12, 2, 7, 5, 12, 17}, {13, 2, 8, 5, 13, 19},
    {15, 3, 9, 6, 15, 21}, {16, 3, 9, 6, 16, 22},
    {17, 3, 10, 7, 17, 24}, {18, 3, 11, 7, 18, 26},
    {20, 4, 12, 8, 20, 28}, {21, 4, 12, 8, 21, 29},
    {22, 4, 13, 9, 22, 31}, {23, 4, 14, 9, 23, 33},
    {25, 5, 15, 10, 25, 35}, {26, 5, 15, 10, 26, 36},
    {27, 5, 16, 11, 27, 38}, {28, 5, 17, 11, 28, 40},
    {30, 6, 18, 12, 30, 42},
};

Tetris* tetris_init(const GameConfig* game_config) {
    Tetris* tetris = (Tetris*)malloc(sizeof(Tetris));
    if (tetris == NULL) return NULL;
    memset(tetris, 0, sizeof(Tetris));

    // Config Defaults
    tetris->config.fps = DEFAULT_FPS;
    tetris->config.gravity = DEFAULT_GRAVITY;
    tetris->config.das = DEFAULT_DAS;
    tetris->config.arr = DEFAULT_ARR;
    tetris->config.soft_drop_gravity = DEFAULT_SOFT_DROP_GRAVITY;
    // Prevent division by zero if fps is weird, though default is 60
    float safe_fps = (float)(tetris->config.fps > 0 ? tetris->config.fps : 60);
    tetris->config.drop_interval = 1.0f / (tetris->config.gravity * safe_fps);
    tetris->config.soft_drop_interval = DEFAULT_SOFT_DROP_GRAVITY; // 通常这里是个倍率或者固定值
    tetris->config.undo_interval = DEFAULT_UNDO_INTERVAL;
    tetris->config.lock_delay = DEFAULT_LOCK_DELAY;
    tetris->config.reset_lock_times_limit = DEFAULT_RESET_LOCK_TIMES_LIMIT;

    // State Init
    tetris->state.lock_times_left = tetris->config.reset_lock_times_limit;
    tetris->state.b2b_count = -1; // -1 means no active B2B streak

    // Game Core Init
    Game* temp_game = game_init(game_config);
    if (temp_game) {
        tetris->game = *temp_game; // Structure copy
        game_free(temp_game);      // Free the pointer, keep the data
    } else {
        free(tetris);
        return NULL;
    }

    return tetris;
}

void tetris_free(Tetris* tetris) {
    if (tetris) free(tetris);
}

Tetris* tetris_copy(const Tetris* src) {
    if (!src) return NULL;
    Tetris* dest = (Tetris*)malloc(sizeof(Tetris));
    if (dest) {
        *dest = *src; // Shallow copy is fine as Game has no external pointers other than what it manages internally
    }
    return dest;
}

int tetris_get_atk(Tetris* tetris) {
    AttackType type = tetris->state.attack_type;
    if (type == ATK_NONE) return 0;

    int total_atk = 0;
    int base_dmg = 0;
    
    // Cap REN for table lookup
    int ren = tetris->game.state.ren;
    if (ren < 0) ren = 0;
    if (ren > 20) ren = 20;

    // Check if B2B is preserved
    // Tetris, T-Spin Mini/Normal (All variants), I-Spin (All variants)
    // Note: B2B rules vary, usually Spin and Quads maintain it.
    bool keeps_b2b = (type == ATK_TETRIS || 
                      (type >= ATK_TSMS && type <= ATK_TST) || 
                      (type >= ATK_ISS && type <= ATK_IST)); // Assuming I-Spin maintains too

    // Lookup Base Damage
    if (type >= ATK_SINGLE && type <= ATK_TST) {
        // Standard ATK_TABLE types: Single(0) ... TST(8)
        int idx = (int)type - (int)ATK_SINGLE;
        
        // Use B2B Table if eligible
        // Eligible types for B2B bonus: Tetris(3) and Spines
        if (tetris->state.b2b_count > -1 && keeps_b2b && type >= ATK_TETRIS) {
             // Map ATK_TETRIS(3) -> 0, TSMS(4) -> 1 ...
             int b2b_idx = (int)type - (int)ATK_TETRIS;
             if (b2b_idx >= 0 && b2b_idx < 6) {
                 base_dmg = ATK_TABLE_B2B1[ren][b2b_idx];
             } else {
                 // Fallback (should not happen with correct bounds)
                 if (idx >= 0 && idx < 9) base_dmg = ATK_TABLE[ren][idx];
             }
        } else {
            // Normal Table
            if (idx >= 0 && idx < 9) {
                base_dmg = ATK_TABLE[ren][idx];
            }
        }
    } 
    // Spines other than T-Spin (I, O, S, Z, L, J)
    // Usually treated as 1/2/3 damage, maybe B2B +1
    else if (type >= ATK_ISS) {
        // Simple logic for non-standard spins if not in table
        // Map *SS -> 1, *SD -> 2, *ST -> 3? 
        // Or just use a simple B2B check
        if (tetris->state.b2b_count > -1) base_dmg = 1; // Weak bonus for exotic spins
    }

    // Add raw B2B streak bonus (optional, some games add +1 for every n B2B)
    // Guideline usually just uses the B2B table columns. 
    // If you have a custom mechanic where high B2B adds more damage:
    // if (!keeps_b2b && tetris->state.b2b_count > 4) total_atk += 1; 

    if (tetris->state.is_pc) total_atk += 10;
    
    total_atk += base_dmg;
    return total_atk;
}

void tetris_receive_garbage_line(Tetris* tetris, int line_count) {
    if (line_count <= 0 || line_count >= BOARD_HEIGHT) {
        return;
    }
    
    Board* board = &tetris->game.board;
    
    // --- 修复：使用正确的 RNG 函数 ---
    uint16_t r = util_rand_next(&tetris->game.state.rng_state);
    uint8_t hole_x = r % BOARD_WIDTH;

    // --- 优化：使用 memmove 替代循环 ---
    // Move existing rows UP (from y=0 to y=line_count)
    // Dest: board->rows[line_count]
    // Src:  board->rows[0]
    // Count: BOARD_HEIGHT - line_count
    memmove(&board->rows[line_count], 
            &board->rows[0], 
            (BOARD_HEIGHT - line_count) * sizeof(uint16_t));

    // Fill bottom rows with garbage
    uint16_t garbage_row = BOARD_ROW_MASK & ~(1 << hole_x);
    for (int y = 0; y < line_count; y++) {
        board->rows[y] = garbage_row;
    }
    
    // --- 新增：垃圾行顶起检测 (Block Out) ---
    // 如果垃圾行顶起导致当前方块重叠，游戏结束
    if (board_piece_overlaps(board, &tetris->game.current_piece)) {
        tetris->game.is_game_over = true;
        tetris->state.is_game_over = true;
    }
}

void tetris_receive_attack(Tetris* tetris, int attack) {
    tetris->state.pending_attack += attack;
}

void tetris_send_garbage_line(Tetris* tetris, int line_count) {
    // Placeholder for network callback
    // if (tetris->callbacks.on_attack) ...
    (void)tetris; // Suppress unused warning
    (void)line_count;
}

void tetris_update_clear_rows(Tetris* tetris) {
    // 1. 获取当前攻击信息 (必须在 game_next_step 之前，因为 next_step 会重置 Last Rotate 状态)
    tetris->state.attack_type = game_get_attack_type(&tetris->game);
    tetris->state.is_pc = game_is_perfect_clear(&tetris->game);
    
    // 2. 更新 B2B 状态
    // 注意：B2B 状态的更新时机。通常：如果这次消除了特殊行，B2B+1；如果消除了普通行，B2B重置；如果不消除，B2B保持。
    bool is_special = (tetris->state.attack_type == ATK_TETRIS || 
                      (tetris->state.attack_type >= ATK_TSMS && tetris->state.attack_type <= ATK_TST) || 
                      (tetris->state.attack_type >= ATK_ISS));
    
    bool is_line_clear = (tetris->state.attack_type != ATK_NONE);

    // 计算伤害时使用 *当前* 的 B2B 状态 (即上一回合留下的状态)
    int attack = tetris_get_atk(tetris);
    
    // 3. 执行核心逻辑 (锁定 -> 消行 -> 更新REN -> 生成新方块)
    game_next_step(&tetris->game);
    
    // 4. 根据结果更新下一回合的 B2B
    if (is_line_clear) {
        if (is_special) {
            tetris->state.b2b_count++;
        } else {
            // 普通消除中断 B2B
            tetris->state.b2b_count = -1;
        }
    }
    // 如果没有消除 (game_next_step 里判断了)，B2B 保持不变 (Tetris Guideline)

    tetris->state.atk_count += attack;

    // 5. 垃圾行抵消逻辑 (Garbage Cancellation)
    if (tetris->game.state.is_last_clear_line) {
        // 这一帧有消除发生，尝试抵消垃圾
        if (tetris->state.pending_attack > 0) {
            if (attack >= tetris->state.pending_attack) {
                // 攻击力完全抵消垃圾，并有剩余
                int overflow = attack - tetris->state.pending_attack;
                tetris->state.pending_attack = 0;
                if (overflow > 0) {
                    tetris_send_garbage_line(tetris, overflow);
                }
            } else {
                // 攻击力不足以抵消所有垃圾
                tetris->state.pending_attack -= attack;
            }
        } else {
            // 无垃圾，直接发送所有攻击
            if (attack > 0) {
                tetris_send_garbage_line(tetris, attack);
            }
        }
    } else {
        // 这一帧没有消除，接收垃圾
        if (tetris->state.pending_attack > 0) {
            // 限制一次接收的行数，避免瞬间暴毙 (例如一次最多8行)
            int incoming = tetris->state.pending_attack;
            if (incoming > 8) incoming = 8;
            
            tetris->state.pending_attack -= incoming;
            tetris_receive_garbage_line(tetris, incoming);
        }
    }

    // Sync Game Over state
    tetris->state.is_game_over = tetris->game.is_game_over;
    tetris->state.is_update_clear_rows_needed = false;
}