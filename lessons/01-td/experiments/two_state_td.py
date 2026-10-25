"""固定策略的两状态 TD(0) 价值预测；只使用 Python 标准功能。"""

alpha = 0.2
gamma = 0.9
v_a = 0.0
v_b = 0.0

for episode in range(1, 51):
    # A -> B: 奖励为 0，使用此刻的 B 估计。
    v_a += alpha * (0 + gamma * v_b - v_a)

    # B -> T: 本回合真正终止，不再自举。
    v_b += alpha * (1 - v_b)

    if episode in (1, 2, 50):
        print(episode, round(v_a, 6), round(v_b, 6))
