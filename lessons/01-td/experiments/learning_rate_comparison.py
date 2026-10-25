"""同一确定性任务、相同预算下，比较首次达到同一精度的回合数。"""

gamma = 0.9
episode_budget = 200
tolerance = 0.01

for alpha in (0.05, 0.2, 0.8):
    v_a = 0.0
    v_b = 0.0
    first_hit = None

    for episode in range(1, episode_budget + 1):
        v_a += alpha * (0 + gamma * v_b - v_a)
        v_b += alpha * (1 - v_b)

        # 真值只用于评估，不参与上面的更新。
        true_a = gamma
        true_b = 1.0
        error = max(abs(v_a - true_a), abs(v_b - true_b))
        if error < tolerance and first_hit is None:
            first_hit = episode

    # None 表示在统一预算内没有达标。
    print(f"alpha={alpha}, first_hit={first_hit}")
