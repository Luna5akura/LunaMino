# ai/test_greedy.py

import numpy as np
import time
import argparse
import sys
import random

# 引入核心游戏逻辑和奖励计算
from .utils import TetrisGame
from .reward import get_reward, calculate_heuristics
from . import config

# 引入 Rich 进行漂亮的输出
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def run_greedy_test(args):
    console = Console()
    
    # 初始化游戏
    game = TetrisGame()
    
    for game_idx in range(args.games):
        seed = int(time.time()) if args.seed < 0 else args.seed + game_idx
        game.reset(seed)
        
        if not args.no_render:
            game.enable_render()
            
        # 统计数据
        stats = {
            'score': 0, 
            'lines': 0, 
            'steps': 0,
            'tetrises': 0
        }
        
        # 奖励函数需要的状态追踪
        episode_stats = {
            'pieces_since_last_garbage': 0,
            'total_garbage_cleared': 0,
            'garbage_clear_events': 0,
            'pieces_used_for_garbage': 0
        }
        
        # 获取初始状态
        board, prev_ctx = game.get_state()
        board = board.copy() # 必须 copy，因为后续模拟会修改
        prev_ctx = prev_ctx.copy()
        prev_metrics = calculate_heuristics(board)
        
        console.rule(f"[bold blue]Greedy Game {game_idx+1} (Seed: {seed})[/bold blue]")
        
        try:
            while True:
                start_time = time.time()
                
                # 渲染延时
                if not args.no_render:

                    game.render()
                    time.sleep(max(0.1, args.speed))
                
                # 1. 获取所有合法动作
                legal_moves = game.get_legal_moves()
                
                if len(legal_moves) == 0:
                    console.print("[bold red]No legal moves! Game Over.[/bold red]")
                    break
                
                # 2. 遍历每一个动作，模拟并计算 Reward
                best_move = None
                best_reward = -float('inf')
                best_info = None
                
                # 用于 Debug 显示前3名的动作
                move_evaluations = []
                
                for i, move in enumerate(legal_moves):
                    # move: [x, y, rotation, landing_height, use_hold]
                    
                    # === 关键：使用 clone() 创建一个沙盒环境进行模拟 ===
                    with game.clone() as sim_game:
                        res = sim_game.step(move[0], move[1], move[2], move[4])
                        
                        next_board_view, next_ctx_view = sim_game.get_state()
                        next_board = next_board_view.copy()
                        next_ctx = next_ctx_view.copy()
                        
                        cur_metrics = calculate_heuristics(next_board)
                        
                        step_result = {
                            'lines_cleared': res[0],
                            'damage_sent': res[1],
                            'attack_type': res[2],
                            'game_over': res[3],
                            'b2b_count': res[4],
                            'combo': res[5]
                        }
                        
                        reward, force_over = get_reward(
                            step_result, cur_metrics, prev_metrics, stats['steps'],
                            context=next_ctx, prev_context=prev_ctx,
                            episode_stats=episode_stats.copy(), 
                            is_training=True 
                        )
                        
                        move_evaluations.append({
                            'move': move,
                            'reward': reward,
                            'metrics': cur_metrics,
                            'lines': res[0]
                        })
                        
                        if reward > best_reward:
                            best_reward = reward
                            best_move = move
                            best_info = step_result

                # 3. 排序并打印调试信息 (前 3 名)
                move_evaluations.sort(key=lambda x: x['reward'], reverse=True)
                
                if stats['steps'] % 10 == 0: # 每10步打印一次详细分析表
                    game.receive_garbage(1)
                    t = Table(title=f"Step {stats['steps']} Greedy Analysis")
                    t.add_column("Rank")
                    t.add_column("Move (x,y,r)")
                    t.add_column("Reward")
                    t.add_column("Lines")
                    t.add_column("Holes")
                    t.add_column("Bump")
                    t.add_column("MaxH")
                    
                    for i in range(min(3, len(move_evaluations))):
                        e = move_evaluations[i]
                        m = e['move']
                        metrics = e['metrics']
                        mv_str = f"x={m[0]}, y={m[1]}, r={m[2]}"
                        t.add_row(
                            str(i+1), 
                            mv_str, 
                            f"[green]{e['reward']:.2f}[/green]", 
                            str(e['lines']),
                            str(metrics['holes']),
                            f"{metrics['bumpiness']:.1f}",
                            str(metrics['max_height'])
                        )
                    console.print(t)
                
                # 4. 在真实环境中执行最好的动作
                if best_move is None:
                    best_move = legal_moves[0]
                
                episode_stats['pieces_since_last_garbage'] += 1
                
                # 执行真实的一步
                res = game.step(best_move[0], best_move[1], best_move[2], best_move[4])
                
                # 5. 更新真实状态
                board_view, ctx_view = game.get_state()
                board = board_view.copy()
                prev_ctx = ctx_view.copy()
                prev_metrics = calculate_heuristics(board)
                
                # =================================================================
                # [新增代码] 打印每一步的 Hole 数量
                # =================================================================
                current_holes = prev_metrics['holes']
                max_height = prev_metrics['max_height']
                console.print(f"Step {stats['steps']+1} > Holes: [bold cyan]{current_holes}[/bold cyan] | MaxHeight: {max_height} | Lines: {res[0]}")
                # =================================================================

                # 统计
                lines = res[0]
                stats['lines'] += lines
                if lines == 4:
                    stats['tetrises'] += 1
                    stats['score'] += 800 * (res[5] + 1)
                elif lines > 0:
                    stats['score'] += 100 * lines * (res[5] + 1)
                
                stats['steps'] += 1
                
                # 计算处理时间 FPS
                elapsed = time.time() - start_time
                # fps = 1.0 / elapsed if elapsed > 0 else 0
                
                if res[3]: # Game Over
                    console.print(f"[bold red]GAME OVER at Step {stats['steps']}[/bold red]")
                    break
            
            # 游戏结束总结
            summary = Table(title=f"Game {game_idx+1} Summary")
            summary.add_column("Metric", style="cyan")
            summary.add_column("Value", style="magenta")
            summary.add_row("Total Lines", str(stats['lines']))
            summary.add_row("Tetrises", str(stats['tetrises']))
            summary.add_row("Total Steps", str(stats['steps']))
            summary.add_row("Final Score", str(stats['score']))
            console.print(summary)
            
            time.sleep(2)

        except KeyboardInterrupt:
            console.print("\n[yellow]Test interrupted by user.[/yellow]")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tetris Greedy Reward Test")
    
    parser.add_argument("--games", type=int, default=3, help="Number of games to play")
    parser.add_argument("--speed", type=float, default=0.02, help="Visualization delay")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--no-render", action="store_true", help="Disable visualization")

    args = parser.parse_args()
    run_greedy_test(args)