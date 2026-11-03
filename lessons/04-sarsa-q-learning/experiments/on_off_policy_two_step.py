"""两步tabular control实验：Sarsa与Q-learning。只需Python标准库。

直接运行打印基线；扩展实验只改EPSILON。每个episode至多两次transition。
此环境为教学自编，所有transition和reward都确定，随机性仅来自action selection。
"""

import random
from statistics import mean


EPSILON = 0.2
EPISODES = 10000
SEEDS = (7, 19, 42, 101, 2026)
GAMMA = 0.9
WINDOW = 1000

# actions使用数字索引：S的0/1=safe/shortcut，B的0/1=good/bad。
ACTION_NAMES = {"S": ("safe", "shortcut"), "B": ("good", "bad")}


def environment_step(state, action):
    if state == "S":
        return (1.0, "T", True) if action == 0 else (0.0, "B", False)
    if state == "B":
        return (2.0, "T", True) if action == 0 else (-10.0, "T", True)
    raise ValueError("只能从非terminal state S或B执行action。")


def choose_action(q_values, epsilon, uniform_number):
    """epsilon概率在全部actions中均匀explore；greedy并列也均分。"""
    best = max(q_values)
    greedy = [a for a, q in enumerate(q_values) if q == best]
    probabilities = [epsilon / len(q_values)] * len(q_values)
    for a in greedy:
        probabilities[a] += (1 - epsilon) / len(greedy)
    cumulative = 0.0
    for a, probability in enumerate(probabilities):
        cumulative += probability
        if uniform_number < cumulative:
            return a
    return len(q_values) - 1


def update(q, visits, state, action, target):
    # 每个state-action pair分别计数，alpha=1/N，不是固定alpha。
    visits[state][action] += 1
    alpha = 1.0 / visits[state][action]
    td_error = q[state][action] - target
    q[state][action] -= alpha * td_error


def train(method, epsilon, random_tape, gamma=GAMMA):
    if method not in ("Sarsa", "Q-learning"):
        raise ValueError("method必须为Sarsa或Q-learning。")
    if not 0 <= epsilon <= 1 or not random_tape:
        raise ValueError("epsilon需要在[0,1]，且至少训练一个episode。")

    q = {"S": [0.0, 0.0], "B": [0.0, 0.0]}
    visits = {"S": [0, 0], "B": [0, 0]}
    returns = []
    steps = 0

    for u_s, u_b in random_tape:
        a_s = choose_action(q["S"], epsilon, u_s)
        reward, next_state, terminal = environment_step("S", a_s)
        steps += 1

        if terminal:
            update(q, visits, "S", a_s, reward)
            returns.append(reward)
            continue

        # 已实际选定的next action，下面会执行同一个a_b。
        a_b = choose_action(q[next_state], epsilon, u_b)
        if method == "Sarsa":
            next_value = q[next_state][a_b]
        else:
            # max只用于target；不把实际a_b强行改成greedy action。
            next_value = max(q[next_state])
        target = reward + gamma * next_value
        update(q, visits, "S", a_s, target)

        # 真正执行刚才选择的a_b；B到T是terminal transition。
        last_reward, _, done = environment_step(next_state, a_b)
        if not done:
            raise RuntimeError("本教学环境的第二步应当terminal。")
        steps += 1
        update(q, visits, next_state, a_b, last_reward)
        returns.append(reward + gamma * last_reward)

    return {"q": q, "visits": visits, "returns": returns, "steps": steps}


def evaluate_greedy(q, gamma=GAMMA):
    """冻结q，不update，不explore；并列时固定选索引较小者。"""
    state, discount, total = "S", 1.0, 0.0
    actions = []
    for _ in range(2):
        action = max(range(len(q[state])), key=lambda a: q[state][a])
        actions.append(ACTION_NAMES[state][action])
        reward, state, terminal = environment_step(state, action)
        total += discount * reward
        if terminal:
            return total, " -> ".join(actions)
        discount *= gamma
    raise RuntimeError("本环境不应超过两步。")


def run_experiment(epsilon=EPSILON, episodes=EPISODES, seeds=SEEDS):
    if not 1 <= episodes <= 100000:
        raise ValueError("教学运行预算为1到100000 episodes。")
    if not seeds:
        raise ValueError("至少需要一个seed。")
    rows = []
    for seed in seeds:
        rng = random.Random(seed)
        # 两算法共享每个episode中S/B的uniform numbers。
        # q不同会改变actions，所以不宣称实际trajectory或returns相同。
        tape = [(rng.random(), rng.random()) for _ in range(episodes)]
        for method in ("Sarsa", "Q-learning"):
            result = train(method, epsilon, tape)
            evaluation, route = evaluate_greedy(result["q"])
            window = min(WINDOW, episodes)
            rows.append({
                "seed": seed, "method": method, "epsilon": epsilon,
                "q": result["q"], "visits": result["visits"],
                "training_last_window_mean_return": mean(result["returns"][-window:]),
                "evaluation_greedy_return": evaluation,
                "evaluation_route": route, "environment_steps": result["steps"],
                "episodes": episodes, "window": window,
            })
    return rows


if __name__ == "__main__":
    print(f"epsilon={EPSILON}, episodes={EPISODES}, gamma={GAMMA}")
    print("train=保留exploration，末窗口平均discounted return；eval=冻结q后的greedy return")
    rows = run_experiment()
    for row in rows:
        q_s = row["q"]["S"]
        print(
            f"seed={row['seed']:4d} {row['method']:10s} "
            f"qS=[{q_s[0]:.4f}, {q_s[1]:.4f}] "
            f"train={row['training_last_window_mean_return']:.4f} "
            f"eval={row['evaluation_greedy_return']:.4f} "
            f"route={row['evaluation_route']} steps={row['environment_steps']}"
        )
    print("\n跨seeds描述性汇总：")
    for method in ("Sarsa", "Q-learning"):
        selected = [row for row in rows if row["method"] == method]
        print(
            method,
            "train_mean=", round(mean(r["training_last_window_mean_return"] for r in selected), 6),
            "eval_mean=", round(mean(r["evaluation_greedy_return"] for r in selected), 6),
        )
