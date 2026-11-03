"""L06 预习：共享 parameters 时，TD target 会跟着 w 一起动。
只用 Python 标准库；每组固定 60 次更新，不会死循环。"""

GAMMA = 0.9
ALPHA = 0.1
UPDATES = 60
START = (1.0, 0.0)  # (w1, w0)，和第05课第5题相同


def v_hat(x, w):
    w1, w0 = w
    return w1 * x + w0


def td_step(w, x, r, x_next, terminal):
    # 1) 用旧 w 算 TD target，这一步当常数
    target = r if terminal else r + GAMMA * v_hat(x_next, w)
    # 2) error = prediction - target
    error = v_hat(x, w) - target
    # 3) gradient = (error * x, error)；4) w_new = w - alpha * gradient
    w1, w0 = w
    return (w1 - ALPHA * error * x, w0 - ALPHA * error)


def run_a():
    print("A. 只反复更新 x=1 -> x_next=2 这一条（r=0，不终止）")
    print(" update       w1        w0  v_hat(1)    target   |error|")
    w = START
    for k in range(UPDATES + 1):
        target = 0 + GAMMA * v_hat(2, w)
        error = v_hat(1, w) - target
        if k in (0, 1, 2, 5, 10, 20, 40, 60):
            print(f"{k:>7} {w[0]:>8.4f} {w[1]:>9.4f} {v_hat(1, w):>9.4f} {target:>9.4f} {abs(error):>9.4f}")
        if k < UPDATES:
            w = td_step(w, 1, 0, 2, False)


def run_b():
    print("\nB. 完整 episode：x=1 -> x=2（r=0），x=2 -> 终止（r=1）")
    print("   每个 episode 更新两次；30 个 episodes，共 60 次更新")
    print(" episode      w1        w0  v_hat(1)  v_hat(2)")
    w = START
    for ep in range(UPDATES // 2 + 1):
        if ep in (0, 1, 2, 5, 10, 20, 30):
            print(f"{ep:>8} {w[0]:>7.4f} {w[1]:>9.4f} {v_hat(1, w):>9.4f} {v_hat(2, w):>9.4f}")
        if ep < UPDATES // 2:
            w = td_step(w, 1, 0, 2, False)
            w = td_step(w, 2, 1, None, True)


run_a()
run_b()
