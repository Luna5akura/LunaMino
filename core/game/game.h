// core/game/game.h

#ifndef GAME_H
#define GAME_H

#include "../piece/piece.h"
#include "../board/board.h"
#include "previews/previews.h"
#include "bag/bag.h"
#include "../../util/util.h"

typedef struct {
    int fps;
    float gravity;
} GameUIConfig;

typedef struct {
    int preview_count;
    Bool is_hold_enabled;
} GameConfig;

typedef struct {
    Bag* bag;
    Previews* previews;
    PieceType hold_piece;
} GameState;
typedef struct {
    GameUIConfig* ui_config;
    GameConfig* config;
    GameState* state;
    Board* board;
    Piece* current_piece;
} Game;

Game* init_game(Bool is_ui_enabled);
Bool try_move_piece(Game* game, MoveAction action);
Bool try_rotate_piece(Game* game, RotationAction action);
int clear_rows(Board* board);
Bool next_piece(Game* game);

#endif