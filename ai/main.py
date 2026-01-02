# main.py
import argparse
import sys
from ai import runner
from ai import config

def main():
    parser = argparse.ArgumentParser(description="Tetris AI Controller")
    
    # 定义两种主模式
    subparsers = parser.add_subparsers(dest='mode', required=True, help="Operation Mode")
    
    # 模式 1: 单线程 GUI (Single + UI)
    # 使用: python main.py gui
    parser_gui = subparsers.add_parser('gui', help="Case 1: Single thread with Game Interface")
    parser_gui.add_argument('--reset', action='store_true', help="Reset training")

    # 模式 2 & 3: 多线程训练 (Multi-thread)
    # 使用: python main.py train (Case 2: Rich)
    # 使用: python main.py train --no-rich (Case 3: No Rich)
    parser_train = subparsers.add_parser('train', help="Case 2/3: Multi-thread Training")
    parser_train.add_argument('--no-rich', action='store_true', help="Case 3: Disable Rich UI (Plain text)")
    parser_train.add_argument('--workers', type=int, default=config.NUM_WORKERS, help="Number of worker processes")
    parser_train.add_argument('--reset', action='store_true', help="Reset training")

    args = parser.parse_args()

    # 逻辑分发
    if args.mode == 'gui':
        # 情况 1: 单线程有界面
        runner.run_gui(reset=args.reset)
    
    elif args.mode == 'train':
        # 情况 2: 多线程 Rich (args.no_rich 为 False)
        # 情况 3: 多线程 无Rich (args.no_rich 为 True)
        use_rich = not args.no_rich
        runner.run_train(reset=args.reset, use_rich=use_rich, workers=args.workers)

if __name__ == "__main__":
    main()