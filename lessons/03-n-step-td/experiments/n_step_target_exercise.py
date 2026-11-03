"""n步目标练习：补全函数后运行，也可复制到Programiz。

rewards非空，已经是本次选取的奖励前缀，按发生顺序排列。
未终止时已收集齐所需步数；提前终止时只包含到终点的奖励。
terminal指任务真正终止。时间上限截断不在本练习范围内。
函数只计算目标，不与环境交互，也不更新价值估计。
"""


def n_step_target(rewards, next_value, terminal, gamma=0.9):
    """返回本次n步目标；实际使用的步数为len(rewards)。"""
    # TODO 1: 把每笔实际奖励按对应折扣累加，第一笔乘gamma**0。
    # TODO 2: 仅在未真正终止时，添加最后状态的折扣价值估计。
    # TODO 3: 返回目标。
    raise NotImplementedError("请先补全n_step_target，再运行四组输入。")


if __name__ == "__main__":
    cases = [
        ("三步未终止", [0.0, 1.0, 2.0], 4.0, False, 0.9),
        ("第三步终止", [0.0, 1.0, 2.0], 100.0, True, 0.9),
        ("第一步终止", [4.0], 100.0, True, 0.9),
        ("只改gamma", [0.0, 1.0, 2.0], 4.0, False, 0.5),
    ]
    try:
        for name, rewards, next_value, terminal, gamma in cases:
            target = n_step_target(rewards, next_value, terminal, gamma)
            print(name, "target=", round(target, 6))
    except NotImplementedError as error:
        print(error)
