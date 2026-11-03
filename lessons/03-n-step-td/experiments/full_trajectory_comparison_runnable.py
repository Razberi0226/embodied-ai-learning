"""第03课完整trajectory实验：已检查的可运行版本。

默认 MODE='check'，只计算四个目标，不运行训练。
阅读并核对后依次使用 MODE='deterministic' 和 MODE='stochastic'。
只依赖Python标准库；不联网、不访问显卡、不写文件。
本教学实验统一在episode结束后用episode开始时的frozen estimates更新。
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
TRUTH = tuple(GAMMA ** (3 - t) for t in range(4))


def n_step_target(rewards, next_value, terminal, gamma=0.9):
    """沿用昨天已检查的函数：累计observed rewards，仅未终止时接后续value estimate。"""
    target = 0.0
    for i, reward in enumerate(rewards):
        target += gamma ** i * reward
    if not terminal:
        target += gamma ** len(rewards) * next_value
    return target


def target_from_trajectory(rewards, old_values, t, n, gamma=0.9):
    """返回从位置t开始的目标；n=None表示MC。

    rewards有T笔奖励，old_values有T个非terminal state的frozen estimates。
    状态编号0到T-1；终点编号T，不在old_values中。
    逐行对应关系：
      1. T=len(rewards)；MC的结束位置k=T，否则k=min(t+n,T)。
      2. 本次observed rewards是rewards[t:k]，不包含下标k。
      3. k==T表示真正终止，此时不能访问old_values[T]。
      4. 未终止时的后续value estimate为old_values[k]。
      5. 调用上方n_step_target；也可独立写等价求和。
    """
    T = len(rewards)
    if n is None:  # None在这里代表MC，要用到真正终点。
        k = T
    else:
        k = min(t + n, T)

    selected_rewards = rewards[t:k]  # 已实际收到的奖励片段。
    terminal = (k == T)
    if terminal:
        next_value = 0.0  # 不能读取old_values[T]，它不在列表中。
    else:
        next_value = old_values[k]

    return n_step_target(selected_rewards, next_value, terminal, gamma)


def rmse(values):
    """与该玩具任务已知true state values比较；不是与某次sample return比较。"""
    return math.sqrt(sum((a-b)**2 for a, b in zip(values, TRUTH)) / 4)


def update_episode(values, rewards, n):
    old = values[:]  # 本episode所有目标使用同一份frozen estimates。
    return [
        old[t] + ALPHA * (
            target_from_trajectory(rewards, old, t, n, GAMMA) - old[t]
        )
        for t in range(4)
    ]


def train(terminal_rewards, n):
    if not 1 <= len(terminal_rewards) <= 500:
        raise ValueError("本练习每种方法的episode数须在1到500之间。")
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
    # 本阶段deterministic reference和stochastic设置使用相同episode预算。
    print("matched deterministic reference：episodes=", STOCHASTIC_EPISODES)
    for name, n in METHODS:
        baseline = train([1.0] * STOCHASTIC_EPISODES, n)
        score = statistics.mean(baseline["errors"][-50:])
        print(" ", name, "末50 episodes平均RMSE=", round(score, 6))
    rows = {name: [] for name, _ in METHODS}
    for seed in SEEDS:
        rng = random.Random(seed)
        # 先生成，再共享。每episode只有终止奖励随机：0或2，各半。
        shared = [float(rng.choice((0, 2)))
                  for _ in range(STOCHASTIC_EPISODES)]
        print("seed=", seed, "episodes=", len(shared),
              "前6个终止奖励=", shared[:6])
        for name, n in METHODS:
            result = train(shared, n)
            # 指标是最后50episode各次RMSE的平均，不是平均估计的RMSE。
            score = statistics.mean(result["errors"][-50:])
            rows[name].append(score)
            print(" ", name, "末50episode平均RMSE=", round(score, 6),
                  "最终A=", round(result["values"][0], 6))
    print("跨种子汇总：每行都是5次运行的描述性统计")
    for name, scores in rows.items():
        print(name, "平均分=", round(statistics.mean(scores), 6),
              "种子间标准差=", round(statistics.pstdev(scores), 6))


def main():
    print("当前模式：", MODE)
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
