# ai/train.py

import os
import time
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from .agent import DQNAgent
from .model import QNet
from .config import *
from .utils import TetrisGame

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "tetris_ai.pth")
BEST_MODEL_FILE = os.path.join(CHECKPOINT_DIR, "best_tetris_ai.pth")
if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)

def run_episode(seed, agent_params):
    game = TetrisGame(seed=seed)
    total_reward = 0
    total_lines = 0
    transitions = []
    local_epsilon = agent_params['epsilon']
    local_q_net = QNet()
    local_q_net.load_state_dict(agent_params['q_net'])
    local_q_net.eval()

    for step_num in range(MAX_STEPS_PER_EPISODE):
        prev_board, prev_ctx = game.get_state()
        moves, previews, ids = game.get_legal_moves_with_previews()
        if len(moves) == 0:
            break

        # Duplicate select_action logic using local_q_net and local_epsilon
        if np.random.rand() < local_epsilon:
            idx = np.random.randint(len(moves))
        else:
            with torch.no_grad():
                post_ctxs = [DQNAgent.approximate_post_ctx(prev_ctx, moves[i], previews[i]) for i in range(len(previews))]
                post_ctx_t = torch.tensor(np.stack(post_ctxs), dtype=torch.float32)
                boards_t = torch.tensor(previews, dtype=torch.float32)
                q_values = local_q_net(boards_t, post_ctx_t).squeeze()
                if q_values.dim() == 0:
                    idx = 0
                else:
                    idx = torch.argmax(q_values).item()
        chosen_move = moves[idx]
        chosen_preview = previews[idx]

        x, y, rot, land, hold = chosen_move
        step_res = game.step(x, y, rot, hold)
        post_board, post_ctx = game.get_state()
        done = step_res[3]
        total_lines += step_res[0]
        reward = DQNAgent.compute_reward(None, step_res, prev_board, post_board, land)  # Use static if moved, or instance None
        next_moves = np.empty((0, 5), dtype=np.int8)
        next_previews = np.empty((0, 20, 10), dtype=np.uint8)
        if not done:
            next_moves, next_previews, _ = game.get_legal_moves_with_previews()
        transitions.append((chosen_preview, post_ctx, reward, next_moves, next_previews, post_ctx, done))
        total_reward += reward
        if done:
            break

    game.close()
    return transitions, total_reward, total_lines

def train():
    agent = DQNAgent()
    try:
        agent.load_checkpoint(CHECKPOINT_FILE)
    except Exception as e:
        print(f"Could not load checkpoint (architecture mismatch?): {e}")
        print("Starting fresh.")
    print("Starting After-State DQN Training...")

    episode = 0
    while episode < MAX_EPISODES:
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            batch_size = min(NUM_THREADS, MAX_EPISODES - episode)
            seeds = [(episode + i + int(time.time())) % 10000 for i in range(batch_size)]
            agent_params = {'epsilon': agent.epsilon, 'q_net': agent.q_net.state_dict()}
            futures = [executor.submit(run_episode, seed, agent_params) for seed in seeds]

            for future in futures:
                trans_list, ep_reward, ep_lines = future.result()
                total_loss = 0
                opt_steps = 0
                for trans in trans_list:
                    agent.replay_buffer.push(*trans)
                    loss = agent.optimize()
                    if loss != 0:
                        total_loss += loss
                        opt_steps += 1
                avg_loss = total_loss / opt_steps if opt_steps > 0 else 0
                print(f"Ep {episode}| R:{ep_reward:.1f} | Lns:{ep_lines} | Loss:{avg_loss:.2f} | Eps:{agent.epsilon:.3f} | Buf:{len(agent.replay_buffer)}")
                if ep_reward > agent.best_reward:
                    agent.best_reward = ep_reward
                    print(f"--> New Best: {ep_reward:.1f}")
                    agent.save_checkpoint(BEST_MODEL_FILE)
                if episode % 50 == 0:
                    agent.save_checkpoint(CHECKPOINT_FILE)
                agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)
                episode += 1
                if episode >= MAX_EPISODES:
                    break

if __name__ == "__main__":
    train()