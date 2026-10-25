"""同一路径和奖励下，改变折扣因子会改变价值的定义。"""

alpha = 0.2

for gamma in (0.9, 0.5):
    v_a = 0.0
    v_b = 0.0

    for episode in range(1, 51):
        v_a += alpha * (0 + gamma * v_b - v_a)
        v_b += alpha * (1 - v_b)

    print(
        f"gamma={gamma}: estimated A={v_a:.6f}, B={v_b:.6f}; "
        f"true A={gamma:.6f}, B=1.000000"
    )
