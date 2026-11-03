"""第04课续学：只改变behavior采样安排，Q-learning target不变。

Python标准库即可；每episode最多两步。核心实验WARM_START=False；
扩展只改成True。初始四个强制采样episodes计入相同总预算。
"""

import random
from statistics import mean

EPISODES = 2000
SEEDS = (7, 19, 42, 101, 2026)
GAMMA = 0.9
DECAY_EPISODES = 1500
WARM_START = False
WINDOW = 500


def epsilon_at(episode_index, schedule):
    if schedule == "fixed":
        return 0.2
    if schedule == "linear":
        progress = min(episode_index / DECAY_EPISODES, 1.0)
        return 0.2 + (0.02 - 0.2) * progress
    if schedule == "fast":
        return 0.2 / (episode_index + 1)
    raise ValueError("schedule应为fixed、linear或fast。")


def choose_action(q_values, epsilon, u):
    best = max(q_values)
    greedy = [a for a, value in enumerate(q_values) if value == best]
    probabilities = [epsilon / len(q_values)] * len(q_values)
    for a in greedy:
        probabilities[a] += (1 - epsilon) / len(greedy)
    cumulative = 0.0
    for a, probability in enumerate(probabilities):
        cumulative += probability
        if u < cumulative:
            return a
    return len(q_values) - 1


def step(state, action):
    # S: 0=safe, 1=shortcut；B: 0=good, 1=bad。
    if state == "S":
        return (1.0, "T", True) if action == 0 else (0.0, "B", False)
    if state == "B":
        return (2.0, "T", True) if action == 0 else (-10.0, "T", True)
    raise ValueError("state应为S或B。")


def update(q, visits, state, action, target):
    visits[state][action] += 1
    alpha = 1.0 / visits[state][action]
    td_error = q[state][action] - target
    q[state][action] -= alpha * td_error


def evaluate_greedy(q):
    # 冻结q，无exploration；并列时固定选索引0。
    a_s = max(range(2), key=lambda a: q["S"][a])
    reward, state, terminal = step("S", a_s)
    if terminal:
        return reward, "safe"
    a_b = max(range(2), key=lambda a: q[state][a])
    last_reward, _, _ = step(state, a_b)
    route = "shortcut -> " + ("good" if a_b == 0 else "bad")
    return reward + GAMMA * last_reward, route


def train(schedule, tape, warm_start=False):
    if warm_start and len(tape) < 4:
        raise ValueError("warm start至少需要4个episodes的预算。")
    q = {"S": [0.0, 0.0], "B": [0.0, 0.0]}
    visits = {"S": [0, 0], "B": [0, 0]}
    returns, checkpoints = [], []
    steps, first_shortcut = 0, None
    # 这四个采样episodes和其后的采样共同使用总预算，不额外赠送经验。
    forced_s = (0, 1, 1, 1)
    forced_b = (None, 0, 1, 0)

    for e, (u_s, u_b) in enumerate(tape):
        epsilon = epsilon_at(e, schedule)
        forced = warm_start and e < 4
        a_s = forced_s[e] if forced else choose_action(q["S"], epsilon, u_s)
        reward, next_state, terminal = step("S", a_s)
        steps += 1
        if terminal:
            update(q, visits, "S", a_s, reward)
            episode_return = reward
        else:
            # 每个schedule始终使用相同Q-learning target。
            target = reward + GAMMA * max(q[next_state])
            update(q, visits, "S", a_s, target)
            a_b = forced_b[e] if forced else choose_action(q[next_state], epsilon, u_b)
            last_reward, _, _ = step(next_state, a_b)
            steps += 1
            update(q, visits, next_state, a_b, last_reward)
            episode_return = reward + GAMMA * last_reward
        returns.append(episode_return)

        if first_shortcut is None and q["S"][1] > q["S"][0]:
            first_shortcut = e + 1
        if e + 1 in (1, 10, 100, 500, 1000, len(tape)):
            value, route = evaluate_greedy(q)
            checkpoints.append({"episode": e + 1, "epsilon": epsilon,
                                "q_S": q["S"][:], "greedy_return": value,
                                "greedy_route": route})

    evaluation, route = evaluate_greedy(q)
    window = min(WINDOW, len(tape))
    return {"schedule": schedule, "warm_start": warm_start,
            "q": q, "visits": visits, "environment_steps": steps,
            "first_greedy_shortcut_episode": first_shortcut,
            "training_last_window_mean": mean(returns[-window:]),
            "greedy_return": evaluation, "greedy_route": route,
            "checkpoints": checkpoints, "episodes": len(tape), "window": window}


def run_experiment(warm_start=WARM_START, episodes=EPISODES, seeds=SEEDS):
    if not 4 <= episodes <= 50000 or not seeds:
        raise ValueError("使用4到50000 episodes和至少一个seed。")
    rows = []
    for seed in seeds:
        rng = random.Random(seed)
        tape = [(rng.random(), rng.random()) for _ in range(episodes)]
        for schedule in ("fixed", "linear", "fast"):
            result = train(schedule, tape, warm_start)
            result["seed"] = seed
            rows.append(result)
    return rows


if __name__ == "__main__":
    print("Q-learning; episodes=", EPISODES, "warm_start=", WARM_START)
    print("各episode编号对应的epsilon：")
    for schedule in ("fixed", "linear", "fast"):
        print(schedule, [(i, round(epsilon_at(i - 1, schedule), 6))
                         for i in (1, 10, 100, 1000, EPISODES)])
    rows = run_experiment()
    print("visits: S=[safe,shortcut], B=[good,bad]")
    for row in rows:
        print(f"seed={row['seed']:4d} {row['schedule']:6s} "
              f"visits={row['visits']} eval={row['greedy_return']:.3f} "
              f"route={row['greedy_route']} steps={row['environment_steps']}")
    print("跨seeds描述性汇总：")
    for schedule in ("fixed", "linear", "fast"):
        selected = [r for r in rows if r["schedule"] == schedule]
        shortcut_count = sum(r["greedy_route"] == "shortcut -> good" for r in selected)
        print(schedule, "greedy_shortcut_runs=", shortcut_count, "/", len(selected),
              "mean_greedy_return=", round(mean(r["greedy_return"] for r in selected), 6))
