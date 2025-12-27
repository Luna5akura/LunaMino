// core/battle/battle.h

#ifndef BATTLE_H
#define BATTLE_H

#include "../tetris/tetris.h"

typedef struct {
    int player_count;
    int seed1;
    int seed2;
} BattleConfig;

typedef struct {
    int score_player1;
    int score_player2;
    Bool is_game_over;
} BattleState;

typedef struct {
    BattleConfig* config;
    BattleState* state;
    Tetris* player1;
    Tetris* player2;
} Battle;

Battle* init_battle(BattleConfig* config);
void run_battle();
void free_battle(Battle* battle);

#endif