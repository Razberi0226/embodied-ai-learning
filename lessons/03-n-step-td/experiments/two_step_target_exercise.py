"""二步 TD 编程练习：补全函数，再在本地或 Programiz 运行。

本练习已收集到两次实际转移；terminal 指第二步到达的状态真正终止。
第一步就终止、或还没收集够两步的情况，将在一般 n 步实现中处理。
函数未补全时会打印提示，不会产生实验结果。
"""


def two_step_update(value, r1, r2, next_value, terminal,
                    gamma=0.9, alpha=0.2):
    """返回 (二步目标, 更新后的当前状态估计)。"""
    # TODO 1: 用两笔实际奖励构成目标；未终止时再接后续状态估计。
    # TODO 2: 使用传入的旧估计 value 计算新估计。
    # TODO 3: 返回 (target, new_value)。
    raise NotImplementedError("请先补全 two_step_update 函数，再运行下面三组输入。")


if __name__ == "__main__":
    cases = [
        ("未终止", (1.0, 0.0, 1.0, 2.0, False), {}),
        ("第二步终止", (1.0, 0.0, 1.0, 100.0, True), {}),
        ("只改 gamma", (1.0, 0.0, 1.0, 2.0, False), {"gamma": 0.5}),
    ]
    try:
        for name, args, kwargs in cases:
            target, new_value = two_step_update(*args, **kwargs)
            print(name, "target=", round(target, 6),
                  "new_value=", round(new_value, 6))
    except NotImplementedError as error:
        print(error)
