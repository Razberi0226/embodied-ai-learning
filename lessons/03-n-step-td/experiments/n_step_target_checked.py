"""n步目标函数：根据截图排错后的参考实现。

本文件只做固定小列表的数值运算，不访问网络、文件或显卡。
rewards是已经收集好的非空奖励前缀；未终止时已收集齐所需步数，
提前终止时只到终点。terminal指最后一步到达的状态真正终止。
该函数不负责环境交互或处理时间上限截断。
"""


def n_step_target(rewards, next_value, terminal, gamma=0.9):
    # 每笔奖励都已经实际收到，终止与否都要计入。
    target = 0.0
    for i, reward in enumerate(rewards):
        target += (gamma ** i) * reward

    # 仅在最后到达的状态未真正终止时，接上后续价值估计。
    if not terminal:
        target += (gamma ** len(rewards)) * next_value

    return target


if __name__ == "__main__":
    cases = [
        ("三步未终止", [0.0, 1.0, 2.0], 4.0, False, 0.9),
        ("第三步终止", [0.0, 1.0, 2.0], 100.0, True, 0.9),
        ("第一步终止", [4.0], 100.0, True, 0.9),
        ("只改gamma", [0.0, 1.0, 2.0], 4.0, False, 0.5),
    ]
    for name, rewards, next_value, terminal, gamma in cases:
        target = n_step_target(rewards, next_value, terminal, gamma)
        print(name, "target=", round(target, 6))
