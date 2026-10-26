"""同一条轨迹上的MC与TD：A -> B -> C -> terminal，奖励0、0、1。

两种方法使用相同的逐回合经历、初始估计、学习率和折扣因子。
该确定性短路径用于理解更新机制，不能据此判断普遍优劣。
"""

alpha = 0.2
gamma = 0.9
td = [0.0, 0.0, 0.0]  # 依次对应A、B、C
mc = [0.0, 0.0, 0.0]

for episode in range(1, 4):
    rewards = [0.0, 0.0, 1.0]  # 本回合实际收到的奖励

    # TD：每一步只用当步奖励和下一状态的当前估计更新
    for i in range(3):
        if i == 2:  # C之后已经终止
            target = rewards[i]
        else:
            target = rewards[i] + gamma * td[i + 1]
        td[i] += alpha * (target - td[i])

    # MC：等回合结束后，从后向前累计实际回报G
    # G只使用本回合奖励，不读取任何下一状态的价值估计
    G = 0.0
    for i in reversed(range(3)):
        G = rewards[i] + gamma * G
        mc[i] += alpha * (G - mc[i])

    print(episode, 'TD:', [round(v, 6) for v in td],
          'MC:', [round(v, 6) for v in mc])
