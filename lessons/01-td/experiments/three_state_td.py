"""学习者独立完成的三状态 TD 扩展整理版。

保留原更新逻辑，额外打印第 3 回合以检查价值估计的传播顺序。
路径 A -> B -> C -> T，奖励依次 0、0、1。
"""

alpha = 0.2
gamma = 0.9
v_a = v_b = v_c = 0.0

for episode in range(1, 51):
    v_a += alpha * (0 + gamma * v_b - v_a)
    v_b += alpha * (0 + gamma * v_c - v_b)
    v_c += alpha * (1 - v_c)

    if episode in (1, 2, 3, 50):
        print(episode, round(v_a, 6), round(v_b, 6), round(v_c, 6))
