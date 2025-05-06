// core/game/game.h

#ifndef GAME_H
#define GAME_H

#include "../piece/piece.h"
#include "../board/board.h"
#include "previews/previews.h"
#include "bag/bag.h"

typedef struct {
    GameConfig* config;
    GameState* state;
    Board* board;
    Piece* current_piece;
} Game;

typedef struct {
    int preview_count;
} GameConfig;

typedef struct {
    Bag* bag;
    Previews* previews;
} GameState;

Game* init_game();
void try_move_piece(Game* game, MoveAction action);
void try_rotate_piece(Game* game, RotationAction action);
int clear_rows(Board* board);
void next_piece(Game* game);

#endif