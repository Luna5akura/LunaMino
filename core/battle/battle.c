// core/battle/battle.c

#include "battle.h"
#include <stdlib.h>

BattleState* init_battle_state() {
    BattleState* state = malloc(sizeof(BattleState));
    return state;
}

Battle* init_battle(BattleConfig* config) {
    Battle* battle = malloc(sizeof(Battle));
    battle->state = init_battle_state();
}
