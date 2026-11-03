"""固定输入下比较三种TD targets；无需第三方库，无长循环。"""


def epsilon_greedy_probs(q_values, epsilon):
    """exploration在所有actions中均匀抽取；greedy并列时均分。"""
    if not q_values or not 0 <= epsilon <= 1:
        raise ValueError("需要非空q_values，且epsilon在[0, 1]内。")
    best_value = max(q_values.values())
    best_actions = [a for a in q_values if q_values[a] == best_value]
    probs = {a: epsilon / len(q_values) for a in q_values}
    for action in best_actions:
        probs[action] += (1 - epsilon) / len(best_actions)
    return probs


def compare_targets(reward, next_q, next_action, policy,
                    terminal=False, gamma=0.9):
    if terminal:
        # 真正terminal时，不读取next_q、next_action或policy。
        return {name: float(reward)
                for name in ["Sarsa", "Expected Sarsa", "Q-learning"]}

    return {
        "Sarsa": reward + gamma * next_q[next_action],
        "Expected Sarsa": reward + gamma * sum(
            policy[a] * next_q[a] for a in next_q
        ),
        "Q-learning": reward + gamma * max(next_q.values()),
    }


def update_from_target(current_q, target, alpha):
    # 按赵老师教材的符号方向：error=current-target，update减alpha*error。
    td_error = current_q - target
    return current_q - alpha * td_error


def show_case(label, next_q, next_action, policy):
    print(label)
    print("policy:", {a: round(p, 6) for a, p in policy.items()})
    targets = compare_targets(0, next_q, next_action, policy)
    for method, target in targets.items():
        # 三个方法分别从相同旧值2独立更新，不串接结果。
        new_q = update_from_target(2, target, 0.2)
        print(f"{method}: target={target:.6f}, new estimate={new_q:.6f}")
    print()


if __name__ == "__main__":
    q = {"L": 4.0, "R": 1.0}
    policy_02 = epsilon_greedy_probs(q, 0.2)
    show_case("基线：epsilon=0.2，实际next action R", q, "R", policy_02)
    show_case("只把实际next action改为L", q, "L", policy_02)

    policy_06 = epsilon_greedy_probs(q, 0.6)
    show_case("只把epsilon改为0.6，实际next action仍为R", q, "R", policy_06)

    print("真正terminal，reward=1:")
    print(compare_targets(1, None, None, None, terminal=True))
