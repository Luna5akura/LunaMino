// core/game/game.h
#ifndef GAME_H
#define GAME_H

#include "../../util/util.h"
#include "../piece/piece.h"
#include "../board/board.h"
#include "previews/previews.h"
#include "bag/bag.h"

// ... (AttackType 枚举保持不变) ...
typedef enum {
    ATK_NONE,
    ATK_SINGLE, ATK_DOUBLE, ATK_TRIPLE, ATK_TETRIS,
    ATK_TSMS, ATK_TSS, ATK_TSMD, ATK_TSD, ATK_TST,
    ATK_ISS, ATK_ISD, ATK_IST,
    ATK_OSS, ATK_OSD,
    ATK_SSS, ATK_SSD, ATK_SST,
    ATK_ZSS, ATK_ZSD, ATK_ZST,
    ATK_JSS, ATK_JSD, ATK_JST,
    ATK_LSS, ATK_LSD, ATK_LST,
    ATK_ERROR,
} AttackType;

typedef struct {
    int preview_count;
    bool is_hold_enabled; // 使用 bool (stdbool.h) 替代 Bool，除非你有特殊定义
    unsigned int seed;
} GameConfig;

typedef struct {
    Bag bag;
    Previews previews;
   
    Piece hold_piece;
    bool has_hold_piece;
    bool can_hold_piece;

    uint32_t rng_state; 
   
    int8_t is_last_rotate; // 优化：int8_t (0-5)
    bool is_last_clear_line;
    int ren; 
} GameState;

typedef struct {
    GameConfig config;
    GameState state;
    Board board;          // 嵌入结构体，内存连续
    Piece current_piece;
    bool is_game_over;
} Game;

// --- API ---
Game* game_init(const GameConfig* config);
void game_free(Game* game);
Game* game_copy(const Game* src);

bool game_try_move(Game* game, MoveAction action);
bool game_try_rotate(Game* game, RotationAction action);
bool game_hold_piece(Game* game);
bool game_next_step(Game* game);

// 辅助查询
int game_get_shadow_height(const Game* game);
AttackType game_get_attack_type(const Game* game);
bool game_is_perfect_clear(const Game* game);

// --- 内联核心逻辑优化 ---

// 优化：使用位运算直接放置，循环减少到 4 次
static inline void place_piece(Board* board, const Piece* p) {
    uint16_t mask = piece_get_mask(p);
    
    // 遍历方块的 4 行
    for (int r = 0; r < 4; r++) {
        // 提取方块当前行的数据 (4 bits)
        // mask 存储格式：行0在高位 (Bits 15-12)
        uint16_t row_data = (mask >> ((3 - r) * 4)) & 0xF;
        
        if (row_data) {
            int y = p->pos[1] - r;
            // 确保 Y 在合法范围内
            if (y >= 0 && y < BOARD_HEIGHT) {
                // 将方块行数据位移到棋盘对应 X 位置
                // 注意：如果 x < 0，需要特殊处理位移
                if (p->pos[0] >= 0) {
                    board->rows[y] |= (row_data << p->pos[0]);
                } else {
                    board->rows[y] |= (row_data >> (-p->pos[0]));
                }
            }
        }
    }
}

// 声明 Board 相关的扩展函数 (实现在 game.c)
bool board_piece_overlaps(const Board* board, const Piece* p);

#endif