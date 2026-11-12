"""第10課：SAC——replay buffer、twin Q、entropy 溫度 α。
只用 Python 標準庫。亂數產生器是自己寫的，所以在任何 Python 環境跑出來都完全一樣。
為了看清楚原理，這裡用表格（每個 state、action 一格）代替神經網路，
而且 policy 直接取 softmax(Q / α)，也就是 SAC 的 actor 想要學到的樣子。
"""
import math

SEEDS = tuple(range(1, 11))
GAMMA = 0.99
LR = 0.1          # 每次更新往 target 走的比例
BATCH = 4         # 每走一步，每個 Q 從 replay buffer 隨機抽 4 筆來更新
TAU = 0.05        # soft target：Q̄ ← Q̄ + τ·(Q − Q̄)。真實 SAC 預設 0.005；這裡問題小，用大一點學得快

# ---------- 核心實驗：高估（maximization bias） ----------
#   A：left → 結束，reward 0；right → 走到 B，reward 0
#   B：N_B 個 action，選哪個都結束，reward 平均 MEAN_B、外加 ±NOISE_B 的均勻雜訊
N_B = 8
MEAN_B = -0.1
NOISE_B = 1.5
CORE_ALPHA = 0.01      # α 很小：policy 幾乎只選估計最高的 action
CORE_EPISODES = 400
CORE_WINDOWS = ((0, 100), (100, 400))   # 前 100 回合、後 300 回合
# 核心實驗只改 MODE：
#   "single"  只有一個 Q，算 target 時直接用它
#   "twin"    兩個 Q，各自從 buffer 抽自己的一批；算 target 時每個 action 取兩者較小的那個（SAC 的做法）
CORE_MODES = ("single", "twin")

# ---------- B 段：放大看高估從哪來（不訓練，只抽樣） ----------
ZOOM_SAMPLES = 5       # 每個 Q 對每個 action 各用 5 筆 reward 取平均
ZOOM_REPEATS = 1000

# ---------- 擴充實驗：陷阱與寶藏，只改 α ----------
#   S：safe → 結束，reward +1；go → 走到 C1，reward 0
#   C1…C4：quit → 結束，reward 0；go → 下一格，reward 0；在 C4 選 go → 拿到寶藏 TREASURE
EXT_ALPHAS = (0.05, 0.3, 1.0, 3.0)
CHAIN = 4
TREASURE = 3.0
REWARD_SCALE = 1.0     # 所有 reward 都乘上這個數
EXT_EPISODES = 300


class RNG:
    """64-bit 線性同餘亂數產生器：u() 回傳 0 到 1 之間的數。"""

    def __init__(self, seed):
        self.state = (seed * 0x9E3779B97F4A7C15 + 1) % 2 ** 64
        for _ in range(5):
            self.u()

    def u(self):
        self.state = (6364136223846793005 * self.state + 1442695040888963407) % 2 ** 64
        return (self.state >> 11) / 2 ** 53

    def pick(self, probs):
        x = self.u()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if x < acc:
                return i
        return len(probs) - 1


def policy(q, alpha):
    """SAC 的 policy：π(a) ∝ exp(Q(a) / α)。"""
    top = max(q)
    e = [math.exp((v - top) / alpha) for v in q]
    total = sum(e)
    return [v / total for v in e]


def entropy(p):
    return -sum(x * math.log(x) for x in p if x > 0)


def soft_value(q, p, alpha):
    """V(s) = Σ π(a)·(Q(s,a) − α·ln π(a))：平均 Q，再加上 α × entropy。"""
    return sum(pa * (qa - alpha * math.log(pa)) for pa, qa in zip(p, q) if pa > 0)


class Agent:
    """表格版的 SAC critic：一個或兩個 Q，各自有 target 版本。"""

    def __init__(self, n_actions, twin, alpha):
        self.n = n_actions                    # 每個 state 有幾個 action
        self.k = 2 if twin else 1
        self.alpha = alpha
        self.Q = [[[0.0] * m for m in n_actions] for _ in range(self.k)]
        self.QT = [[[0.0] * m for m in n_actions] for _ in range(self.k)]
        self.buffer = []                      # replay buffer：(s, a, r, s')，s' = None 表示結束

    def q_min(self, table, s):
        if self.k == 1:
            return table[0][s]
        return [min(table[0][s][a], table[1][s][a]) for a in range(self.n[s])]

    def act_probs(self, s):
        return policy(self.q_min(self.Q, s), self.alpha)

    def learn(self, rng):
        cache = {}

        def v_target(s):                      # target 用 target Q（取 min）＋目前的 policy
            if s not in cache:
                cache[s] = soft_value(self.q_min(self.QT, s), self.act_probs(s), self.alpha)
            return cache[s]

        for i in range(self.k):               # 每個 Q 各抽自己的一批
            for _ in range(BATCH):
                s, a, r, s2 = self.buffer[int(rng.u() * len(self.buffer))]
                y = r if s2 is None else r + GAMMA * v_target(s2)
                self.Q[i][s][a] += LR * (y - self.Q[i][s][a])
        for i in range(self.k):               # soft target update
            for s in range(len(self.n)):
                for a in range(self.n[s]):
                    self.QT[i][s][a] += TAU * (self.Q[i][s][a] - self.QT[i][s][a])


def run_core(mode, seed):
    rng = RNG(seed)
    agent = Agent([2, N_B], mode == "twin", CORE_ALPHA)
    went_right, v_b, q_right = [], [], []
    for _ in range(CORE_EPISODES):
        a = rng.pick(agent.act_probs(0))
        went_right.append(a)
        if a == 0:
            agent.buffer.append((0, 0, 0.0, None))
            agent.learn(rng)
        else:
            agent.buffer.append((0, 1, 0.0, 1))
            agent.learn(rng)
            b = rng.pick(agent.act_probs(1))
            agent.buffer.append((1, b, MEAN_B + NOISE_B * (2 * rng.u() - 1), None))
            agent.learn(rng)
        qb = agent.q_min(agent.Q, 1)
        v_b.append(soft_value(qb, policy(qb, CORE_ALPHA), CORE_ALPHA))
        q_right.append(agent.q_min(agent.Q, 0)[1])
    return went_right, v_b, q_right


def zoom(seed=1):
    """每個 action 的真值都是 MEAN_B。兩個 Q 各用自己的 5 筆 reward 取平均，看取 max 之後會怎樣。"""
    rng = RNG(seed)

    def estimate():
        return [sum(MEAN_B + NOISE_B * (2 * rng.u() - 1) for _ in range(ZOOM_SAMPLES)) / ZOOM_SAMPLES
                for _ in range(N_B)]

    q1, q2 = estimate(), estimate()
    example = (q1, q2)
    sums = [0.0, 0.0, 0.0]
    for rep in range(ZOOM_REPEATS):
        if rep > 0:
            q1, q2 = estimate(), estimate()
        sums[0] += sum(q1) / N_B
        sums[1] += max(q1)
        sums[2] += max(min(x, y) for x, y in zip(q1, q2))
    return example, [s / ZOOM_REPEATS for s in sums]


def run_ext(alpha, seed):
    rng = RNG(seed)
    agent = Agent([2] * (CHAIN + 1), True, alpha)
    returns, found = [], 0
    for _ in range(EXT_EPISODES):
        s, total = 0, 0.0
        while s is not None:
            a = rng.pick(agent.act_probs(s))
            if s == 0:
                r, s2 = (1.0, None) if a == 0 else (0.0, 1)
            elif a == 0:
                r, s2 = 0.0, None
            elif s == CHAIN:
                r, s2 = TREASURE, None
                found += 1
            else:
                r, s2 = 0.0, s + 1
            r *= REWARD_SCALE
            agent.buffer.append((s, a, r, s2))
            agent.learn(rng)
            total += r
            s = s2
        returns.append(total)
    return returns, found, agent.act_probs(0)


def ratio2(count, n):
    """count / n 四捨五入到小數點後兩位（只用整數運算，任何 Python 環境都印得一樣）。"""
    h = (200 * count + n) // (2 * n)
    return f"{h // 100}.{h % 100:02d}"


def main():
    v_true = MEAN_B + CORE_ALPHA * math.log(N_B)
    print("A. 核心實驗：只改 Q 的數量（數字是 10 個 seed 的平均）")
    print(f"   真值：V(B) = {v_true:+.3f}，Q(A,right) = {GAMMA * v_true:+.3f}，Q(A,left) = 0 → 幾乎都該走 left")
    for mode in CORE_MODES:
        runs = [run_core(mode, s) for s in SEEDS]
        print(f"\nMODE = {mode}")
        print("     回合    | 走 right 的比例   V(B) 估計   Q(A,right) 估計")
        for lo, hi in CORE_WINDOWS:
            n = len(SEEDS) * (hi - lo)
            right = ratio2(sum(sum(r[0][lo:hi]) for r in runs), n)
            vb = sum(sum(r[1][lo:hi]) for r in runs) / n
            qr = sum(sum(r[2][lo:hi]) for r in runs) / n
            print(f"  {lo + 1:>4}–{hi:<4}  |      {right}        {vb:+7.3f}      {qr:+7.3f}")
        lo, hi = CORE_WINDOWS[-1]
        print(f"  每個 seed 第 {lo + 1}–{hi} 回合走 right 的比例：" + ", ".join(ratio2(sum(r[0][lo:hi]), hi - lo) for r in runs))

    print("\nB. 放大看：高估從哪來（不訓練，只抽樣）")
    print(f"   B 的 {N_B} 個 action 真值都是 {MEAN_B}；每個 Q 對每個 action 各用 {ZOOM_SAMPLES} 筆 reward 取平均")
    (q1, q2), avg = zoom()
    print("   第 1 次抽樣：")
    print("     Q1        = " + ", ".join(f"{x:+.2f}" for x in q1) + f"   → 最大 {max(q1):+.2f}")
    print("     Q2        = " + ", ".join(f"{x:+.2f}" for x in q2))
    m = [min(x, y) for x, y in zip(q1, q2)]
    print("     min(Q1,Q2)= " + ", ".join(f"{x:+.2f}" for x in m) + f"   → 最大 {max(m):+.2f}")
    print(f"   重複 {ZOOM_REPEATS} 次的平均：Q1 各 action 的平均 {avg[0]:+.3f}；"
          f"max Q1 {avg[1]:+.3f}；max min(Q1,Q2) {avg[2]:+.3f}；真值 {MEAN_B:+.3f}")

    print("\nC. 擴充實驗：只改 α（陷阱與寶藏）")
    print(f"   safe 拿 {1.0 * REWARD_SCALE:g}；走完 {CHAIN} 格拿寶藏 {TREASURE * REWARD_SCALE:g}；"
          f"中途 quit 拿 0。twin Q，{EXT_EPISODES} 回合，{len(SEEDS)} 個 seed")
    print("       α | 拿過寶藏的 seed   最後100回合平均 reward（最小的 seed）   最後 π(go|S)")
    for alpha in EXT_ALPHAS:
        res = [run_ext(alpha, s) for s in SEEDS]
        totals = [sum(r[0][-100:]) for r in res]          # 每個 seed 最後 100 回合的 reward 總和
        found = sum(1 for r in res if r[1] > 0)
        go = sum(r[2][1] for r in res) / len(res)
        print(f"  {alpha:>6g} |      {found:>2} / {len(SEEDS)}            "
              f"{sum(totals) / (100 * len(res)):7.3f}  ({min(totals) / 100:6.2f})                  {go:.2f}")


if __name__ == "__main__":
    main()
