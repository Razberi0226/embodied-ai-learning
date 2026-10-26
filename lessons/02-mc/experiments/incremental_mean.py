"""MC 实际回报：比较增量样本平均与固定学习率，只使用标准库。"""

import math
import random


GAMMA = 0.9
EPISODES = 1000
SEEDS = (7, 19, 42)
ALPHAS = (0.05, 0.2, 0.8)
CHECKPOINTS = (10, 100, 1000)


def generate_returns(seed, episodes):
    """A -> B -> C -> T；奖励 0、0、R，R 独立等概率取 0 或 2。"""
    rng = random.Random(seed)
    return [GAMMA ** 2 * rng.choice((0.0, 2.0)) for _ in range(episodes)]


def estimate_same_returns(returns, alpha):
    """所有方法从 0 开始，读取完全相同的一串实际回报。"""
    sample_mean = 0.0
    fixed_alpha_value = 0.0
    mean_history = []
    fixed_history = []
    for n, target in enumerate(returns, start=1):
        sample_mean += (target - sample_mean) / n
        fixed_alpha_value += alpha * (target - fixed_alpha_value)
        mean_history.append(sample_mean)
        fixed_history.append(fixed_alpha_value)
    return mean_history, fixed_history


def main():
    true_value = GAMMA ** 2 * 1.0  # E[R] = 1；只用于检查，不用于更新。
    print(f"gamma={GAMMA}, episodes={EPISODES}, true_value={true_value:.6f}")
    print("seed alpha episode sample_mean fixed_alpha")
    for seed in SEEDS:
        returns = generate_returns(seed, EPISODES)
        direct_average = sum(returns) / len(returns)
        for alpha in ALPHAS:
            mean_history, fixed_history = estimate_same_returns(returns, alpha)
            for n in CHECKPOINTS:
                print(seed, alpha, n,
                      f"{mean_history[n - 1]:.6f}",
                      f"{fixed_history[n - 1]:.6f}")
            matches = math.isclose(mean_history[-1], direct_average,
                                   rel_tol=1e-12, abs_tol=1e-12)
            print("matches_full_average:", matches)
            if not matches:
                raise AssertionError("Incremental mean disagrees with direct average")
            if seed == 7 and alpha == 0.2:
                print("last_5_returns:", [round(x, 6) for x in returns[-5:]])
                print("last_5_means:", [round(x, 6) for x in mean_history[-5:]])
                print("last_5_fixed:", [round(x, 6) for x in fixed_history[-5:]])


if __name__ == "__main__":
    main()
