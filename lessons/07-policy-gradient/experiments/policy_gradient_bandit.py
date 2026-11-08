"""第07課：policy gradient——直接學 policy（REINFORCE）。
只用 Python 標準庫。亂數產生器是自己寫的，所以在任何 Python 環境跑出來都完全一樣。
環境：一個 state、兩個 action（L、R），選一個就結束（每個 episode 只有一步）。
"""
import math

ALPHA = 0.1
STEPS = 400
SEEDS = (1, 2, 3, 4, 5)
MARKS = (0, 25, 50, 100, 200, 400)

# 擴充實驗只改這個：每次拿到的 reward 都加上 OFFSET（最好的 action 不會因此改變）
EXT_OFFSETS = (0.0, 10.0)

# L：reward 永遠是 1。R：一半機率 4、一半機率 0，平均 2。所以 R 比較好。
NAMES = ("L", "R")
MEAN_REWARD = (1.0, 2.0)   # 只有「精確梯度」那一組會用到；REINFORCE 不知道這兩個數


class RNG:
    """64-bit 線性同餘亂數產生器：u() 回傳 0 到 1 之間的數。"""

    def __init__(self, seed):
        self.state = (seed * 0x9E3779B97F4A7C15 + 1) % 2 ** 64
        for _ in range(5):
            self.u()

    def u(self):
        self.state = (6364136223846793005 * self.state + 1442695040888963407) % 2 ** 64
        return (self.state >> 11) / 2 ** 53


def policy(theta):
    """softmax：把兩個 preference θ 變成兩個機率。"""
    top = max(theta)
    e = [math.exp(t - top) for t in theta]
    total = sum(e)
    return [v / total for v in e]


def sample_reward(a, rng):
    if a == 0:
        return 1.0
    return 4.0 if rng.u() < 0.5 else 0.0


def grad_log_pi(a, p):
    """softmax 的 ∇ln π(a)：選到的那格是 1−π，其他格是 −π。"""
    return [(1.0 if i == a else 0.0) - p[i] for i in range(len(p))]


def exact_gradient(p):
    """∇J 的第 i 格 = π(i) × (第 i 個 action 的平均 reward − 整體平均 J)。"""
    j = sum(p[i] * MEAN_REWARD[i] for i in range(len(p)))
    return [p[i] * (MEAN_REWARD[i] - j) for i in range(len(p))]


def run_exact():
    theta = [0.0, 0.0]
    record = {}
    for k in range(STEPS + 1):
        p = policy(theta)
        if k in MARKS:
            record[k] = p[1]
        g = exact_gradient(p)
        theta = [theta[i] + ALPHA * g[i] for i in range(2)]
    return record


def run_reinforce(seed, offset):
    rng = RNG(seed)
    theta = [0.0, 0.0]
    record = {}
    for k in range(STEPS + 1):
        p = policy(theta)
        if k in MARKS:
            record[k] = p[1]
        if k == STEPS:
            break
        a = 1 if rng.u() < p[1] else 0          # 照目前的 policy 抽 action
        r = sample_reward(a, rng) + offset       # 只看到這一次的 reward
        g = grad_log_pi(a, p)
        theta = [theta[i] + ALPHA * r * g[i] for i in range(2)]
    return record


def show_table(title, rows_by_seed, exact=None):
    print(f"\n{title}")
    head = " step"
    if exact is not None:
        head += "    精確"
    head += "".join(f"  seed{s:<3}" for s in SEEDS) + "    平均  最大-最小"
    print(head)
    for k in MARKS:
        vals = [rows_by_seed[s][k] for s in SEEDS]
        line = f"{k:>5}"
        if exact is not None:
            line += f"  {exact[k]:>6.3f}"
        line += "".join(f"  {v:>7.3f}" for v in vals)
        line += f"  {sum(vals) / len(vals):>6.3f}  {max(vals) - min(vals):>9.3f}"
        print(line)
    stuck = sum(1 for s in SEEDS if rows_by_seed[s][MARKS[-1]] < 0.1)
    print(f"最後 π(R) < 0.1（卡在 L）的 seed 數：{stuck} / {len(SEEDS)}")


def main():
    print("A. 從 θ = (0, 0) 出發，REINFORCE 一次更新的三種可能（α = 0.1，OFFSET = 0）")
    p = policy([0.0, 0.0])
    cases = ((0, 1.0, 0.5), (1, 4.0, 0.25), (1, 0.0, 0.25))   # (action, reward, 發生機率)
    expected = [0.0, 0.0]
    for a, r, prob in cases:
        g = grad_log_pi(a, p)
        delta = [ALPHA * r * g[i] + 0.0 for i in range(2)]
        new_p = policy(delta)
        expected = [expected[i] + prob * delta[i] for i in range(2)]
        print(f"選到 {NAMES[a]}，r = {r:g}（機率 {prob}）：∇ln π = ({g[0]:+.2f}, {g[1]:+.2f})，"
              f"θ 變化 = ({delta[0]:+.3f}, {delta[1]:+.3f})，π(R) 變成 {new_p[1]:.3f}")
    g = exact_gradient(p)
    print(f"三種情況按機率平均的 θ 變化 = ({expected[0]:+.4f}, {expected[1]:+.4f})")
    print(f"α × 精確梯度             = ({ALPHA * g[0]:+.4f}, {ALPHA * g[1]:+.4f})")

    print("\nB. 核心實驗：只改「梯度怎麼來」（表中數字是 π(R)，越接近 1 越好）")
    exact = run_exact()
    rows = {s: run_reinforce(s, 0.0) for s in SEEDS}
    show_table("精確梯度（知道平均 reward） vs REINFORCE（只靠抽樣），OFFSET = 0", rows, exact)

    print("\nC. 擴充實驗：REINFORCE，只改 OFFSET")
    for offset in EXT_OFFSETS:
        rows = {s: run_reinforce(s, offset) for s in SEEDS}
        show_table(f"OFFSET = {offset:g}", rows)


if __name__ == "__main__":
    main()
