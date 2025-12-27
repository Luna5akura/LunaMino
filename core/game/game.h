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
    ATK_SINGLE,
    ATK_DOUBLE,
    ATK_TRIPLE,
    ATK_TETRIS,
    ATK_TSMS,
    ATK_TSS,
    ATK_TSMD,
    ATK_TSD,
    ATK_TST,
    ATK_ISS,
    ATK_ISD,
    ATK_IST,
    ATK_OSS,
    ATK_OSD,
    ATK_SSS,
    ATK_SSD,
    ATK_SST,
    ATK_ZSS,
    ATK_ZSD,
    ATK_ZST,
    ATK_JSS,
    ATK_JSD,
    ATK_JST,
    ATK_LSS,
    ATK_LSD,
    ATK_LST,
    ATK_ERROR,
} AttackType;

typedef struct {
    int preview_count;
    Bool is_hold_enabled;
    int seed;
} GameConfig;

typedef struct {
    Bag* bag;
    Previews* previews;
    Piece* hold_piece;
    Bool can_hold_piece;
    int is_last_rotate; // 0: not rotate, 1-4: rotate, 5: rotate-5
    Bool is_last_clear_line;
    int ren;
} GameState;

typedef struct {
    GameConfig* config;
    GameState* state;
    Board* board;
    Piece* current_piece;
} Game;

// --- [新增/修改] 公开这些初始化函数的声明，以便 ai_reset_game 调用 ---
GameConfig* init_game_config();
void free_game_config(GameConfig* config);
GameState* init_game_state(GameConfig* config);
void free_game_state(GameState* state);
// -------------------------------------------------------------

Game* init_game();
void free_game(Game* game);
Game* copy_game(Game* game);

Bool try_move_piece(Game* game, MoveAction action);
Bool try_rotate_piece(Game* game, RotationAction action);
int clear_rows(Board* board);
Bool next_piece(Game* game);
Bool try_hold_piece(Game* game);
Bool is_grounded(Game* game);
Bool lock_piece(Game* game);
Bool spawn_piece(Game* game);
int get_shadow_height(Game* game);
AttackType get_attack_type(Game* game);
Bool is_t_triple_corner(Game* game);
Bool is_perfect_clear(Game* game);
void update_ren(Game* game);
Bool is_overlapping(Board* board, Piece* piece);
Bool hard_drop(Piece* piece, Board* board);

#endif