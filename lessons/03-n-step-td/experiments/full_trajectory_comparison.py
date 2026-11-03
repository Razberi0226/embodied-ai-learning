"""第03课完整轨迹实验：只需补全 target_from_trajectory。

默认 MODE='check'，只计算四个目标，不运行训练。
补全并核对后依次使用 MODE='deterministic' 和 MODE='stochastic'。
只依赖Python标准库；不联网、不访问显卡、不写文件。
本教学实验统一在回合结束后用回合开始时的冻结估计更新。
"""

import math
import random
import statistics

MODE = "check"  # check / deterministic / stochastic
GAMMA = 0.9
ALPHA = 0.2
DETERMINISTIC_EPISODES = 80
STOCHASTIC_EPISODES = 300
SEEDS = (7, 19, 42, 101, 2026)
METHODS = (("n=1", 1), ("n=2", 2), ("n=3", 3), ("MC", None))
TRUTH = (0.729, 0.81, 0.9, 1.0)


def n_step_target(rewards, next_value, terminal, gamma=0.9):
    """沿用昨天已检查的函数：累计实际奖励，仅未终止时接后续估计。"""
    target = 0.0
    for i, reward in enumerate(rewards):
        target += gamma ** i * reward
    if not terminal:
        target += gamma ** len(rewards) * next_value
    return target


def target_from_trajectory(rewards, old_values, t, n, gamma=0.9):
    """返回从位置t开始的目标；n=None表示MC。

    rewards有T笔奖励，old_values有T个非终止状态的冻结估计。
    状态编号0到T-1；终点编号T，不在old_values中。
    hints：
      1. T=len(rewards)；MC的结束位置k=T，否则k=min(t+n,T)。
      2. 本次实际奖励是rewards[t:k]，不包含下标k。
      3. k==T表示真正终止，此时不能访问old_values[T]。
      4. 未终止时的后续估计为old_values[k]。
      5. 调用上方n_step_target；也可独立写等价求和。
    """
    raise NotImplementedError("请先补全 target_from_trajectory，再运行四组检查。")


def rmse(values):
    """与该玩具任务已知理论真值比较；不是与某次样本回报比较。"""
    return math.sqrt(sum((a-b)**2 for a, b in zip(values, TRUTH)) / 4)


def update_episode(values, rewards, n):
    old = values[:]  # 本回合所有目标使用同一份冻结估计。
    return [
        old[t] + ALPHA * (
            target_from_trajectory(rewards, old, t, n, GAMMA) - old[t]
        )
        for t in range(4)
    ]


def train(terminal_rewards, n):
    if not 1 <= len(terminal_rewards) <= 500:
        raise ValueError("本练习每种方法的回合数须在1到500之间。")
    values = [0.0] * 4
    errors = []
    first_positive_a = None
    first_below_threshold = None
    snapshots = {}
    for episode, final_reward in enumerate(terminal_rewards, 1):
        values = update_episode(values, [0.0, 0.0, 0.0, final_reward], n)
        error = rmse(values)
        errors.append(error)
        if first_positive_a is None and values[0] > 0:
            first_positive_a = episode
        if first_below_threshold is None and error < 0.02:
            first_below_threshold = episode
        if episode in (1, 2, 4, len(terminal_rewards)):
            snapshots[episode] = [round(x, 6) for x in values]
    return {
        "values": values,
        "errors": errors,
        "snapshots": snapshots,
        "first_positive_a": first_positive_a,
        "first_below_threshold": first_below_threshold,
    }


def check_targets():
    rewards = [0.0, 0.0, 0.0, 1.0]
    old = [1.0, 2.0, 3.0, 4.0]
    cases = (("从A看两步", 0, 2), ("从C看三步", 2, 3),
             ("从D看一步", 3, 1), ("从A用MC", 0, None))
    for name, t, n in cases:
        result = target_from_trajectory(rewards, old, t, n, GAMMA)
        print(name, "target=", round(result, 6))


def deterministic_experiment():
    # 每个方法复用完全相同的确定奖励序列。
    rewards = [1.0] * DETERMINISTIC_EPISODES
    print("确定奖励：gamma=", GAMMA, "alpha=", ALPHA,
          "episodes=", len(rewards))
    for name, n in METHODS:
        result = train(rewards, n)
        print(name, "A首次为正=", result["first_positive_a"],
              "RMSE首次<0.02=", result["first_below_threshold"],
              "最终RMSE=", round(result["errors"][-1], 6))
        print("  状态顺序[A,B,C,D]：", result["snapshots"])


def stochastic_experiment():
    if not 1 <= len(SEEDS) <= 5:
        raise ValueError("本练习使用1到5个种子。")
    rows = {name: [] for name, _ in METHODS}
    for seed in SEEDS:
        rng = random.Random(seed)
        # 先生成，再共享。每回合只有终止奖励随机：0或2，各半。
        shared = [float(rng.choice((0, 2)))
                  for _ in range(STOCHASTIC_EPISODES)]
        print("seed=", seed, "episodes=", len(shared),
              "前6个终止奖励=", shared[:6])
        for name, n in METHODS:
            result = train(shared, n)
            # 指标是最后50回合各次RMSE的平均，不是平均估计的RMSE。
            score = statistics.mean(result["errors"][-50:])
            rows[name].append(score)
            print(" ", name, "末50回合平均RMSE=", round(score, 6),
                  "最终A=", round(result["values"][0], 6))
    print("跨种子汇总：每行都是5次运行的描述性统计")
    for name, scores in rows.items():
        print(name, "平均分=", round(statistics.mean(scores), 6),
              "种子间标准差=", round(statistics.pstdev(scores), 6))


def main():
    print("这是固定小规模的教学实验；默认只检查四个目标。")
    try:
        if MODE == "check":
            check_targets()
        elif MODE == "deterministic":
            deterministic_experiment()
        elif MODE == "stochastic":
            stochastic_experiment()
        else:
            raise ValueError("MODE只能是check、deterministic或stochastic。")
    except NotImplementedError as error:
        print(error)


if __name__ == "__main__":
    main()
