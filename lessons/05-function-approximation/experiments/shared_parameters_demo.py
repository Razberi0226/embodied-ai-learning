"""只演示value表示与共享参数；没有训练循环，不执行TD update。"""


def value_estimate(x, weights):
    slope, intercept = weights
    return slope * x + intercept


if __name__ == "__main__":
    states = (0, 1, 2)
    old_w = (2.0, 1.0)
    # 人为给定对照参数，下一正式课再推导怎样学习参数。
    new_w = (2.0, 2.0)
    table = {x: value_estimate(x, old_w) for x in states}
    changed_table = table.copy()
    changed_table[1] = 4.0

    print("x | before | table只改x=1 | function改共享intercept")
    for x in states:
        print(x, table[x], changed_table[x], value_estimate(x, new_w))

    query = 1.5
    print("未单独存入table的x:", query)
    print("table:", table.get(query, "未存储"))
    print("旧function estimate:", value_estimate(query, old_w))
    print("新function estimate:", value_estimate(query, new_w))
    print("这些是estimates；演示没有提供各state的真实value。")
