"""第09課：PPO——同一批資料用好幾次，靠 clip 防止走太遠；GAE 用 λ 取捨。
只用 Python 標準庫。亂數產生器是自己寫的，所以在任何 Python 環境跑出來都完全一樣。
"""
import math

SEEDS = tuple(range(1, 11))   # 10 個 seed：PPO 要防的那種崩壞不常發生，seed 少了看不到

# ---------- 核心實驗：兩個 action 的 bandit（和第07課相同） ----------
#   L：reward 永遠 1；R：一半機率 4、一半機率 0（平均 2）。R 比較好。
ITERATIONS = 30        # 收集資料、更新，重複 30 輪
BATCH = 4              # 每輪只收集 4 筆資料（故意很少，讓運氣的影響看得出來）
EPOCHS = 10            # 同一批資料重複用 10 次
LR = 0.5               # 每個 epoch 的步長
CLIP_EPS = 0.2         # clip 範圍：ratio 只能在 0.8 到 1.2 之間發揮作用

# 核心實驗只改 MODE：
#   "one_step"  每批資料只用一次（第07、08課的做法）
#   "no_clip"   每批資料用 10 次，用 ratio 修正，但不 clip
#   "clip"      每批資料用 10 次，用 ratio 修正，並且 clip（PPO）
CORE_MODES = ("one_step", "no_clip", "clip")
CORE_MARKS = (0, 5, 10, 20, 30)

# ---------- 擴充實驗：GAE 的 λ ----------
EXT_LAMBDAS = (0.0, 0.5, 0.9, 0.95, 1.0)
CRITIC_QUALITY = 0.5   # 1 = critic 完全正確；0 = critic 分不出兩條路
GAMMA = 0.99
CHAIN = 30             # 選完之後還要走 30 步
NOISE = 1.0            # 每一步 reward 的雜訊大小（均勻分佈 ±NOISE）
EXT_BATCH = 16         # 每次用 16 個 episode 估一次梯度
EXT_REPEATS = 300      # 重複估 300 次，看估計值的平均與分散


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
    top = max(theta)
    e = [math.exp(t - top) for t in theta]
    total = sum(e)
    return [v / total for v in e]


def bandit_reward(a, rng):
    if a == 0:
        return 1.0
    return 4.0 if rng.u() < 0.5 else 0.0


def collect(theta, rng):
    """照目前（舊）的 policy 收集一批資料；advantage = reward − 這批的平均。"""
    p_old = policy(theta)
    batch = []
    for _ in range(BATCH):
        a = 1 if rng.u() < p_old[1] else 0
        batch.append((a, bandit_reward(a, rng)))
    mean = sum(r for _, r in batch) / BATCH
    return p_old, [(a, r, r - mean) for a, r in batch]


def update(theta, p_old, batch, mode, record_ratio=None):
    epochs = 1 if mode == "one_step" else EPOCHS
    for _ in range(epochs):
        p = policy(theta)
        g = [0.0, 0.0]
        for a, _, adv in batch:
            ratio = p[a] / p_old[a]                       # 新機率 / 收集時的機率
            if mode == "clip":
                if (adv > 0 and ratio > 1 + CLIP_EPS) or (adv < 0 and ratio < 1 - CLIP_EPS):
                    continue                              # 這筆已經推夠了，不再推
            for i in range(2):
                grad_log = (1.0 if i == a else 0.0) - p[i]
                g[i] += adv * ratio * grad_log / BATCH    # ratio × advantage × ∇ln π
        theta = [theta[i] + LR * g[i] for i in range(2)]
        if record_ratio is not None:
            new_p = policy(theta)
            record_ratio.append((new_p[0] / p_old[0], new_p[1]))
    return theta


def run_core(mode, seed):
    rng = RNG(seed)
    theta = [0.0, 0.0]
    history = [policy(theta)[1]]
    for _ in range(ITERATIONS):
        p_old, batch = collect(theta, rng)
        theta = update(theta, p_old, batch, mode)
        history.append(policy(theta)[1])
    return history


def collapse_example(seed=6, iteration=18):
    """重現 no_clip 在某一輪之前的狀態，再用同一批資料比較 no_clip 和 clip。"""
    rng = RNG(seed)
    theta = [0.0, 0.0]
    for it in range(1, iteration + 1):
        p_old, batch = collect(theta, rng)
        if it == iteration:
            return theta, p_old, batch
        theta = update(theta, p_old, batch, "no_clip")


# ---------- 擴充實驗用的環境 ----------
#   s0 選 R 或 L（policy 固定一半一半，這裡只比較梯度「估得準不準」，不訓練）
#   選了之後各自走 30 步：R 這條每步平均 0.1 分，L 這條每步平均 0 分，每步都有雜訊
MU = {"R": 0.1, "L": 0.0}


def true_value(chain, t):
    """第 t 步（1…CHAIN）所在 state 的真實 value。"""
    return MU[chain] * (1 - GAMMA ** (CHAIN - t + 1)) / (1 - GAMMA)


def critic_value(chain, t):
    """不完美的 critic：在兩條路的平均值與真實值之間，按 CRITIC_QUALITY 取中間。"""
    if t > CHAIN:
        return 0.0
    avg = 0.5 * (true_value("R", t) + true_value("L", t))
    return avg + CRITIC_QUALITY * (true_value(chain, t) - avg)


V0 = GAMMA * 0.5 * (true_value("R", 1) + true_value("L", 1))     # s0 的真實 value
TRUE_ADV_R = GAMMA * true_value("R", 1) - V0                       # 選 R 的真實 advantage
TRUE_GRAD = 0.5 * TRUE_ADV_R                                       # θ_R 方向的真實梯度


def critic_table():
    """先把 critic 對每條路、每一步的估計算好：vhat[chain][t]，t = 0 是 s0。"""
    table = {}
    for chain in ("R", "L"):
        row = [critic_value(chain, t) for t in range(CHAIN + 2)]
        row[0] = V0
        table[chain] = row
    return table


def gae_at_start(vhat, rewards, lam):
    """從最後一步往回算：A_t = δ_t + γλ·A_{t+1}，回傳 s0 的 advantage。"""
    adv = 0.0
    for t in range(CHAIN, -1, -1):
        delta = rewards[t] + GAMMA * vhat[t + 1] - vhat[t]
        adv = delta + GAMMA * lam * adv
    return adv


def estimate_gradient(lam, table, seed=1):
    rng = RNG(seed)
    estimates = []
    for _ in range(EXT_REPEATS):
        g = 0.0
        for _ in range(EXT_BATCH):
            chain = "R" if rng.u() < 0.5 else "L"
            rewards = [MU[chain] + NOISE * (2 * rng.u() - 1) for _ in range(CHAIN + 1)]
            rewards[0] = NOISE * (2 * rng.u() - 1)     # s0 這一步本身沒有差別，只有雜訊
            g += gae_at_start(table[chain], rewards, lam) * ((1.0 if chain == "R" else 0.0) - 0.5)
        estimates.append(g / EXT_BATCH)
    mean = sum(estimates) / len(estimates)
    sd = (sum((x - mean) ** 2 for x in estimates) / len(estimates)) ** 0.5
    return mean, sd


def main():
    print("A. 核心實驗：只改更新方式（兩個 action 的 bandit，數字是 π(R)，越接近 1 越好）")
    print(f"   每輪 {BATCH} 筆資料、{ITERATIONS} 輪；no_clip 與 clip 每批用 {EPOCHS} 次；{len(SEEDS)} 個 seed")
    for mode in CORE_MODES:
        runs = {s: run_core(mode, s) for s in SEEDS}
        print(f"\nMODE = {mode}")
        print("   輪數 |   平均   最小值")
        for k in CORE_MARKS:
            vals = [runs[s][k] for s in SEEDS]
            print(f"  {k:>5} | {sum(vals) / len(vals):6.3f}  {min(vals):7.3f}")
        worst = [min(runs[s][i] - runs[s][i - 1] for i in range(1, ITERATIONS + 1)) for s in SEEDS]
        collapsed = sum(1 for s in SEEDS if runs[s][-1] < 0.1)
        print("  每個 seed 單輪最大的下降：" + ", ".join(f"{w:+.2f}" for w in worst))
        print(f"  最後 π(R) < 0.1 的 seed：{collapsed} / {len(SEEDS)}")

    print("\nB. 放大看：seed 6 第 18 輪，同一個起點、同一批資料")
    theta, p_old, batch = collapse_example()
    print(f"   收集時 π(R) = {p_old[1]:.4f}、π(L) = {p_old[0]:.4f}")
    print("   這批資料（action, reward, advantage）：" + ", ".join(
        f"({'R' if a == 1 else 'L'}, {r:g}, {adv:+.2f})" for a, r, adv in batch))
    for mode in ("no_clip", "clip"):
        trace = []
        update(theta, p_old, batch, mode, record_ratio=trace)
        print(f"   {mode:>7}：每個 epoch 之後的 ratio(L) = " + ", ".join(f"{x:.1f}" for x, _ in trace))
        print(f"            10 個 epoch 之後 π(R) = {trace[-1][1]:.4f}")

    print("\nC. 擴充實驗：只改 GAE 的 λ（梯度估計準不準）")
    print(f"   critic 品質 = {CRITIC_QUALITY}，γ = {GAMMA}，選完後再走 {CHAIN} 步；真實梯度 = {TRUE_GRAD:.3f}")
    print("      λ |  估計平均    偏差   分散(sd)   總誤差(RMSE)")
    table = critic_table()
    for lam in EXT_LAMBDAS:
        mean, sd = estimate_gradient(lam, table)
        bias = mean - TRUE_GRAD
        rmse = (bias ** 2 + sd ** 2) ** 0.5
        print(f"  {lam:>5} |  {mean:8.3f}  {bias:+7.3f}  {sd:9.3f}  {rmse:13.3f}")


if __name__ == "__main__":
    main()
