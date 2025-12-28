// core/game/game.h
#ifndef GAME_H
#define GAME_H
#include "../../util/util.h"
#include "../piece/piece.h"
#include "../board/board.h"
#include "previews/previews.h"
#include "bag/bag.h"
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
    Bool is_hold_enabled;
    unsigned int seed;
} GameConfig;
typedef struct {
    Bag bag;
    Previews previews;
   
    Piece hold_piece;
    Bool has_hold_piece;
    Bool can_hold_piece;
   
    int is_last_rotate; // 0: none, 1-5: normal kicks, for 180 separate but unified
    Bool is_last_clear_line;
    int ren; // Combo count
} GameState;
// 主游戏对象
// 优化：单一大对象，内存连续
typedef struct {
    GameConfig config;
    GameState state;
    Board board;
    Piece current_piece;
    Bool is_game_over; // 新增：方便快速判断游戏结束
} Game;
// --- API ---
// 初始化游戏 (只进行一次 malloc)
Game* game_init(const GameConfig* config);
// 释放游戏
void game_free(Game* game);
// 极速复制游戏 (用于 AI 推演)
// 由于内存连续，这等同于 memcpy
Game* game_copy(const Game* src);
// 核心逻辑
Bool game_try_move(Game* game, MoveAction action);
Bool game_try_rotate(Game* game, RotationAction action);
Bool game_hold_piece(Game* game);
// 推进游戏进程 (锁定 -> 消行 -> 生成下一个)
// 返回 TRUE 表示游戏结束
Bool game_next_step(Game* game);
// 辅助查询
int game_get_shadow_height(const Game* game); // 此时不需要修改 game，加 const
AttackType game_get_attack_type(const Game* game); // 需要根据当前状态判断
Bool game_is_perfect_clear(const Game* game);


static void place_piece(Board* board, const Piece* p) {
    uint16_t mask = piece_get_mask(p);
    for (int dy = 0; dy < 4; dy++) {
        for (int dx = 0; dx < 4; dx++) {
            if (mask & (1u << (15 - (dy * 4 + dx)))) {
                int bx = p->x + dx;
                int by = p->y - dy;
                if (bx >= 0 && bx < board->width && by >= 0 && by < board->height) {
                    board_set_cell(board, bx, by, 1);
                }
            }
        }
    }
}
int clear_rows(Board* board);
Bool board_piece_overlaps(const Board* board, const Piece* p);

#endif