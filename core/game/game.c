// core/game/game.c

#include "game.h"
#include "../../util/util.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


const int NORMAL_PIECE_NORMAL_SRS[4][2][5][2] = {
    { // UP
        { {0, 0}, {-1, 0}, {-1 ,1}, {0, -2}, {-1, -2} }, // CW
        { {0, 0}, {1, 0}, {1, 1}, {0, -2}, {1, -2} }, // CCW
    },
    { // RIGHT
        { {0, 0}, {1, 0}, {1, -1}, {0, 2}, {1, 2} }, // CW
        { {0, 0}, {1, 0}, {1, -1}, {0, 2}, {1, 2} }, // CCW
    },
    { // DOWN
        { {0, 0}, {1, 0}, {1, 1}, {0, -2}, {1, -2} }, // CW
        { {0, 0}, {-1, 0}, {-1, 1}, {0, -2}, {-1, -2} }, // CCW
    },
    { // LEFT
        { {0, 0}, {-1, 0}, {-1, -1}, {0, 2}, {-1, 2} }, // CW
        { {0, 0}, {-1, 0}, {-1, -1}, {0, 2}, {-1, 2} }, // CCW
    }
};

const int I_PIECE_NORMAL_SRS[4][2][5][2] = {
    { // UP
        { {0, 0}, {-2, 0}, {1, 0}, {-2, -1}, {1, 2} }, // CW
        { {0, 0}, {-1, 0}, {2, 0}, {-1, 2}, {2, -1} }, // CCW
    },
    { // RIGHT
        { {0, 0}, {-1, 0}, {2, 0}, {-1, 2}, {2, -1} }, // CW
        { {0, 0}, {2, 0}, {-1, 0}, {2, 1}, {-1, -2} }, // CCW
    },
    { // DOWN
        { {0, 0}, {2, 0}, {-1, 0}, {2, 1}, {-1, -2} }, // CW
        { {0, 0}, {1, 0}, {-2, 0}, {1, -2}, {-2, 1} }, // CCW
    },
    { // LEFT
        { {0, 0}, {1, 0}, {-2, 0}, {1, -2}, {-2, 1} }, // CW
        { {0, 0}, {-2, 0}, {1, 0}, {-2, -1}, {1, 2} }, // CCW
    }
};

const int NORMAL_PIECE_180_SRS[4][6][2] = {
    { {0, 0}, {0, 1}, {1, 1}, {-1, 1}, {1, 0}, {-1, 0} }, // UP
    { {0, 0}, {1, 0}, {1, 2}, {1, 1}, {0, 2}, {0, 1} }, // RIGHT
    { {0, 0}, {0, -1}, {-1, -1}, {1, -1}, {-1, 0}, {1, 0} }, // DOWN
    { {0, 0}, {-1, 0}, {-1, 2}, {-1, 1}, {0, 2}, {0, 1} }, // LEFT
};

const int I_PIECE_180_SRS[4][2][2] = {
    { {0, 0}, {0, -1} }, // UP
    { {0, 0}, {1, 0} }, // RIGHT
    { {0, 0}, {0, 1} }, // DOWN
    { {0, 0}, {-1, 0} }, // LEFT
};

GameConfig* init_game_config() {
    GameConfig* config = malloc(sizeof(GameConfig));
    config->preview_count = 5;
    return config;
}

GameState* init_game_state(GameConfig* config) {
    GameState* state = malloc(sizeof(GameState));
    state->bag = init_bag();

    Previews* previews = init_previews(config->preview_count);
    previews->previews[0] = state->bag->sequence[0];
    for (int i = 1; i < config->preview_count; i++) {
        previews->previews[i] = bag_next_piece(state->bag);
    }
    state->previews = previews;
}

Game* init_game() {
    Game* game = malloc(sizeof(Game));

    GameConfig* config = init_game_config();
    game->config = config;

    GameState* state = init_game_state(config);
    game->state = state;

    Board* board = init_board();
    game->board = board;

    Piece* current_piece = init_piece(state->bag->sequence[config->preview_count]);
    current_piece->x = 3;
    current_piece->y = 21;
    current_piece->rotation = (Rotation)0;
    game->current_piece = current_piece;

    

    return game;
}

Bool is_overlapping(Board* board, Piece* piece) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (piece->shape[i][j] == 0) continue;

            int x = piece->x + j;
            int y = piece->y - i;

            if (x < 0 || x >= board->width || y < 0 || y >= board->height + 3) return TRUE; 
            if (board->state[x][y] != 0) return TRUE;
        }
    }
    return FALSE;
}

void hard_drop(Piece* piece, Board* board) {
    while (!is_overlapping(board, piece)) {
        piece->y--;
    }
    piece->y++;
}

void try_move_piece(Game* game, MoveAction action) {
    Piece* current_piece = game->current_piece;
    printf("try_move_piece: %d, position: (%d, %d), rotation: %d\n", action, current_piece->x, current_piece->y, (int)current_piece->rotation);
    Board* board = game->board;

    // hard drop
    if (action == MOVE_HARD_DROP) {
        hard_drop(current_piece, board);
        return;
    }

    // not hard drop
    int original_x = current_piece->x;
    int original_y = current_piece->y;

    move_piece(current_piece, action);

    if (is_overlapping(board, current_piece)) {
        current_piece->x = original_x;
        current_piece->y = original_y;
        printf("OVERLAPPING\n");
    }
}  


Bool try_displace_piece(Game* game, const int direction[2]) {
    Piece* current_piece = game->current_piece;
    Board* board = game->board;
    int original_x = current_piece->x;
    int original_y = current_piece->y;

    displace_piece(current_piece, direction);

    if (is_overlapping(board, current_piece)) {
        current_piece->x = original_x;
        current_piece->y = original_y;
        return FALSE;
    }
    return TRUE;
}

void try_rotate_piece_normal(Game* game, RotationAction action) {
    Piece* current_piece = game->current_piece;
    Rotation original_rotation = current_piece->rotation;

    int original_shape[4][4];
    memcpy(current_piece->shape, original_shape, sizeof(current_piece->shape));

    rotate_piece(current_piece, action);
    // not I piece
    if (current_piece->type != I_PIECE) {
        for (int i = 0; i < 5; i++) {
            Bool is_successful = try_displace_piece(
                game,
                NORMAL_PIECE_NORMAL_SRS[(int)current_piece->rotation][(int)action][i]
            );
            if (is_successful) {
                return;
            }
        }
        current_piece->rotation = original_rotation;
        memcpy(original_shape, current_piece->shape, sizeof(current_piece->shape));
    }
    // I piece
    else {
        for (int i = 0; i < 5; i++) {
            Bool is_successful = try_displace_piece(
                game, 
                I_PIECE_NORMAL_SRS[(int)current_piece->rotation][(int)action][i]
            );
            if (is_successful) {
                return;
            }
        }
        current_piece->rotation = original_rotation;
        memcpy(original_shape, current_piece->shape, sizeof(current_piece->shape));
    }
}

void try_rotate_piece_180(Game* game, RotationAction action) {
    Piece* current_piece = game->current_piece;
    Rotation original_rotation = current_piece->rotation;

    int original_shape[4][4];
    memcpy(current_piece->shape, original_shape, sizeof(current_piece->shape));

    rotate_piece(current_piece, action);
    // not I piece
    if (current_piece->type != I_PIECE) {
        for (int i = 0; i < 6; i++) {
            Bool is_successful = try_displace_piece(
                game, 
                NORMAL_PIECE_180_SRS[(int)current_piece->rotation][i]
            );
            if (is_successful) {
                return;
            }
        }
        current_piece->rotation = original_rotation;
        memcpy(original_shape, current_piece->shape, sizeof(current_piece->shape));
    }
    // I piece
    else {
        for (int i = 0; i < 2; i++) {
            Bool is_successful = try_displace_piece(
                game,
                I_PIECE_180_SRS[(int)current_piece->rotation][i]
            );
            if (is_successful) {
                return;
            }
        }
        current_piece->rotation = original_rotation;
        memcpy(original_shape, current_piece->shape, sizeof(current_piece->shape));
    }
}

void try_rotate_piece(Game* game, RotationAction action) {
    if (action == ROTATE_180) {
        try_rotate_piece_180(game, action);
    }
    else {
        try_rotate_piece_normal(game, action);
    }
}

void lock_piece(Game* game) {
    Board* board = game->board;
    Piece* current_piece = game->current_piece;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (current_piece->shape[i][j] == 0) continue;

            int x = current_piece->x + j;
            int y = current_piece->y - i;
            
            board->state[x][y] = (int)current_piece->type + 1;
        }
    }
}

int clear_rows(Board* board) {
    int num_rows_cleared = 0;
    for (int y = board->height - 1; y >= 0; y--) {
        Bool is_row_full = TRUE;
        for (int x = 0; x < board->width; x++) {
            if (board->state[x][y] == 0) {
                is_row_full = FALSE;
                break;
            }
        }
        if (is_row_full) {
            num_rows_cleared++;
            for (int yy = y; yy <= board->height - 1; yy++) {
                for (int x = 0; x < board->width; x++) {
                    board->state[x][yy] = board->state[x][yy + 1];
                }
            }
            for (int x = 0; x < board->width; x++) {
                board->state[x][board->height - 1] = 0;
            }
        }
    }
    return num_rows_cleared;
}

void spawn_piece(Game* game) {
    Piece* current_piece = game->current_piece;

    PieceType type = bag_next_piece(game->state->bag);
    Piece* new_piece = init_piece(type);
    new_piece->x = 3;
    new_piece->y = 21;
    game->current_piece = new_piece;
    free(current_piece);
}

void next_piece(Game* game) {
    lock_piece(game);
    spawn_piece(game);
}
