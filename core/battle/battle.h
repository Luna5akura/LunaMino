// core/battle/battle.h

#ifndef BATTLE_H
#define BATTLE_H

#include "../tetris/tetris.h"

typedef struct {

} BattleConfig;

typedef struct {
    
} BattleState;

typedef struct {
    BattleState* state;
    Tetris* players[];
} Battle;

#endif