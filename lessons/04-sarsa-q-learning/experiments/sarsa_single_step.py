"""action values和Sarsa单次update演示。

只使用Python标准库，固定几个小输入；不访问网络、文件或显卡。
这不是完整Sarsa training loop，也没有控制机器人。
"""


def state_value_from_action_values(action_values, policy):
    """在本例中按policy对给定的action values加权。"""
    return sum(policy[a] * action_values[a] for a in policy)


def sarsa_update(q_current, reward, next_q, terminal,
                 gamma=0.9, alpha=0.2):
    """返回TD target、TD error与new estimate。

    next_q来自policy实际选中的next action。
    terminal=True时，next_q可传None，函数不会读取它。
    TD error采用赵老师教材的current estimate减TD target方向。
    """
    target = reward
    if not terminal:
        target += gamma * next_q
    td_error = q_current - target
    new_q = q_current - alpha * td_error
    return target, td_error, new_q


if __name__ == "__main__":
    action_values = {"close": 2.0, "align": 6.0}
    policy = {"close": 0.25, "align": 0.75}
    print("state value =", state_value_from_action_values(action_values, policy))
    print("greedy action =", max(action_values, key=action_values.get))
    for name, next_q, terminal in (
        ("nonterminal", 4.0, False),
        ("terminal", None, True),
    ):
        target, error, updated = sarsa_update(3.0, 1.0, next_q, terminal)
        print(name, "TD target =", round(target, 6),
              "TD error =", round(error, 6),
              "new estimate =", round(updated, 6))
