# Tetris AI (AlphaZero-Tetris) 技术文档

## 1. 系统架构概述

本系统是一个基于 **AlphaZero 方法论** 的高性能俄罗斯方块 AI。它采用了 **C++ (底层引擎)** 与 **Python (PyTorch 深度学习)** 的混合架构，旨在解决高达 $10^{12}$ 状态复杂度的落点预测问题。

*   **游戏引擎 (`libtetris.so`)**: 负责核心规则、状态维护、碰撞检测及合法动作生成。
*   **接口层 (`ai/utils.py`)**: 使用 `ctypes` 和 `numpy` 实现内存零拷贝（Zero-copy）交互。
*   **决策模型 (`ai/model.py`)**: 深度残差网络 (Deep ResNet)，负责评估局面价值和策略概率。
*   **训练系统 (`ai/unified_trainer.py`)**: 支持单/多进程的 Self-Play 训练循环。

---

## 2. 核心神经网络 (Deep ResNet)

为了处理复杂的俄罗斯方块局面（如 T-Spin 构造、堆叠地形分析），模型采用了加深加宽的 **ResNet** 架构。

### 2.1 网络结构概览
*   **类型**: 双头（Dual-Headed）卷积神经网络。
*   **深度**: **10 个残差块 (Residual Blocks)**。
*   **宽度**: **128 通道 (Channels)**。
*   **参数量**: 约 5M ~ 10M。

### 2.2 详细层级流
1.  **输入层 (Input Stem)**:
    *   `Conv2d(1 -> 128, 3x3)` + `BN` + `ReLU`。
    *   将单通道的二值化棋盘映射到高维特征空间。

2.  **骨干网络 (Backbone - Residual Tower)**:
    *   包含 10 个标准的 `ResidualBlock`。
    *   每个 Block 包含两层 `Conv3x3` (维持 128 通道)，通过跳跃连接（Skip Connection）防止梯度消失，提取深层抽象特征（如空洞形状、潜在消除机会）。

3.  **特征融合 (Feature Fusion)**:
    *   除了图像特征，模型还引入了 **Context (11维)** 向量。
    *   Context 经过全连接层映射为 64 维特征。
    *   在进入输出头之前，图像特征被 Flatten，并与 Context 特征进行 **Concat (拼接)**。

4.  **输出头 (Heads)**:
    *   **策略头 (Policy Head)**:
        *   结构: `Conv1x1(32ch)` -> `Flatten` -> `Concat Context` -> `FC(1024)` -> `Dropout(0.3)` -> `FC(2304)`.
        *   输出: 2304 维向量 (Logits)，代表每个 Macro Action 的未归一化概率。
    *   **价值头 (Value Head)**:
        *   结构: `Conv1x1(4ch)` -> `Flatten` -> `Concat Context` -> `FC(256)` -> `Dropout(0.3)` -> `FC(1)` -> `Tanh`.
        *   输出: $[-1, 1]$ 的标量，代表当前局面的胜率/得分期望。

---

## 3. 输入表示 (State Representation)
<tab>
模型接收两部分输入，旨在最大限度地保留局面信息并剔除冗余干扰。

### 3.1 二值化游戏面板 (Binary Board)
*   **Tensor Shape**: `[Batch, 1, 20, 10]`
*   **数据类型**: `Float32` (值为 0.0 或 1.0)
*   **含义**:
    *   `1.0`: 该位置有方块（忽略颜色/类型）。
    *   `0.0`: 该位置为空。
*   **预处理**: 行序倒置 (`[::-1]`)，使得索引 0 对应底部，索引 19 对应顶部。

### 3.2 上下文向量 (Context Input)
*   **Tensor Shape**: `[Batch, 11]`
*   **内容**:
    1.  **Current Piece**: 当前方块 ID。
    2.  **Hold Piece**: 暂存方块 ID。
    3.  **Next Queue (5)**: 接下来 5 个方块的 ID。
    4.  **B2B / Combo**: 连击与 Back-to-Back 状态。
    5.  **Garbage**: 底部即将上涨的垃圾行层数。

---

## 4. 动作空间与编码

模型不输出原子操作（如左移一次），而是直接输出**宏动作（Macro Action / Final Placement）**，极大提升了搜索效率。

*   **动作空间大小**: **2304**。
*   **动作构成**: Unique ID 由以下四个因素决定：
    1.  **Hold**: 是否使用暂存 (0/1)。
    2.  **Rotation**: 旋转状态 (0~3)。
    3.  **Location X**: 落点横坐标 (映射到 0~11)。
    4.  **Landing Height Y**: 落点纵坐标 (0~23, 用于区分同列不同层高的嵌入)。
*   **稀疏性**: 虽然空间有 2304 维，但每一步通常只有 **20~100** 个合法动作（Legal Moves）。

---

## 5. MCTS 搜索与推理流程

系统在推理阶段使用 **Batch MCTS**（批量蒙特卡洛树搜索）。

1.  **扩展 (Expansion)**:
    *   游戏到达某个叶子节点，获取当前合法动作列表。
    *   将状态放入 Batch 队列。
2.  **评估 (Evaluation)**:
    *   当 Batch 攒满（或超时），将一批状态送入 GPU 上的 Deep ResNet。
    *   模型输出 `P` (Policy Logits) 和 `V` (Value)。
3.  **掩码处理 (Action Masking)**:
    *   **关键步骤**: 根据 C++ 引擎提供的合法动作 ID，构建掩码。
    *   将 Policy Logits 中非法动作对应的值设为 $-\infty$。
    *   执行 Softmax，得到合法的先验概率分布。
4.  **模拟 (Simulation)**:
    *   在树上根据 UCB 公式选择路径，直到叶子节点。
    *   引入 **Virtual Loss** 惩罚，防止同一 Batch 内的多条线程走重复路径，增加探索性。

---

## 6. 训练策略 (Training Pipeline)

训练采用自我对弈 (Self-Play) 产生数据，不断迭代更新网络。

### 6.1 数据流
1.  **采集**: Worker 进程使用 MCTS 进行对弈，生成 `(Board, Context, MCTS_Probabilities, Reward)` 元组。
2.  **存储**: 存入 `NumpyReplayBuffer`，使用 `int8` 压缩存储 Board 以节省内存。
3.  **采样**: 训练器从 Buffer 中随机采样 Batch。

### 6.2 损失函数 (Masked Loss)
针对动作空间的稀疏性，训练过程进行了特殊优化：

*   **策略损失 (Policy Loss)**: **Masked Softmax Cross Entropy**
    *   不强迫网络学习“非法动作概率为0”（这浪费了网络容量）。
    *   在计算 Loss 前，根据 Target（MCTS概率）中的非零项构建掩码，将预测 Logits 中的非法项屏蔽。
    *   网络只需专注于**在合法动作之间进行排序**。
*   **价值损失 (Value Loss)**: **MSE Loss**
    *   预测值与最终归一化得分（或胜负）的均方误差。

---

## 7. 接口函数说明 (`ai/utils.py`)

Python 与 C++ 的交互核心。

*   `get_state()`:
    *   **输入**: 无。
    *   **输出**: `(binary_board, context)`。
    *   **功能**: 读取底层内存，二值化棋盘，组装 Context。
*   `get_legal_moves_with_ids()`:
    *   **输出**: `(moves_params, action_ids)`。
    *   **功能**: 返回所有合法落点的物理参数（x, y, rot...）以及对应的全局唯一 ID（用于索引神经网络输出）。
*   `step(x, y, rot, hold)`:
    *   **功能**: 在 C++ 引擎中执行一步，并返回消除行数、是否结束等信息。