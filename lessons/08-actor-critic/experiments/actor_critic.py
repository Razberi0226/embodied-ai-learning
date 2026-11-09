"""第08課：actor-critic——讓 critic 學出 baseline。
只用 Python 標準庫。亂數產生器是自己寫的，所以在任何 Python 環境跑出來都完全一樣。
"""
import math

ALPHA_ACTOR = 0.1     # actor（policy 參數 θ）的步長
ALPHA_CRITIC = 0.1    # critic（V）的步長
GAMMA = 0.9
SEEDS = (1, 2, 3, 4, 5)

# 核心實驗只改 BASELINE：
#   "none"      不減 baseline（第07課的 REINFORCE）
#   "shared"    所有 state 共用一個 baseline（全部 reward 的移動平均）
#   "per_state" 每個 state 各自一個 V(s)，也就是 critic
CORE_BASELINES = ("none", "shared", "per_state")
CORE_EPISODES = 800
CORE_MARKS = (0, 100, 200, 400, 800)

# 擴充實驗只改 critic 的步長（A2C，兩步 MDP）
EXT_CRITIC_ALPHAS = (0.0, 0.1, 0.5)
EXT_EPISODES = 2000
EXT_MARKS = (0, 100, 200, 500, 1000, 2000)


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


def grad_log_pi(a, p):
    return [(1.0 if i == a else 0.0) - p[i] for i in range(len(p))]


def actor_step(theta, a, p, advantage):
    g = grad_log_pi(a, p)
    return [theta[i] + ALPHA_ACTOR * advantage * g[i] for i in range(len(theta))]


# ---------- 核心實驗的環境：兩個 state 的 bandit ----------
# 每個 episode 隨機出現在 X 或 Y（各一半），選 L 或 R 一次就結束。
#   Y：L 永遠 1 分；R 一半機率 4、一半機率 0（平均 2）
#   X：每個 reward 都多 10 分（L 11 分；R 14 或 10 分，平均 12）
# 兩個 state 都是 R 比較好。
EXTRA = {"X": 10.0, "Y": 0.0}


def bandit_reward(s, a, rng):
    if a == 0:
        return 1.0 + EXTRA[s]
    return (4.0 if rng.u() < 0.5 else 0.0) + EXTRA[s]


def run_core(baseline, seed):
    rng = RNG(seed)
    theta = {"X": [0.0, 0.0], "Y": [0.0, 0.0]}
    v = {"X": 0.0, "Y": 0.0}       # per_state 用：critic 的 V(s)
    shared_b = 0.0                 # shared 用：一個共用的數
    record = {}
    for ep in range(CORE_EPISODES + 1):
        if ep in CORE_MARKS:
            record[ep] = (policy(theta["X"])[1], policy(theta["Y"])[1])
        if ep == CORE_EPISODES:
            break
        s = "X" if rng.u() < 0.5 else "Y"
        p = policy(theta[s])
        a = 1 if rng.u() < p[1] else 0
        r = bandit_reward(s, a, rng)
        if baseline == "none":
            advantage = r
        elif baseline == "shared":
            advantage = r - shared_b
            shared_b += ALPHA_CRITIC * (r - shared_b)
        else:
            # 只有一步就終止，所以 TD target 就是 r
            advantage = r - v[s]
            v[s] += ALPHA_CRITIC * (r - v[s])
        theta[s] = actor_step(theta[s], a, p, advantage)
    return record, v, shared_b


# ---------- 擴充實驗的環境：第04課的兩步 MDP ----------
#   S --safe（reward 1）--> 終止
#   S --shortcut（reward 0）--> B --good（reward 2）--> 終止
#                               B --bad（reward −10）--> 終止
# 最佳路線是 shortcut → good（從 S 看 return 0.9×2 = 1.8，比 safe 的 1 好）。


def mdp_step(s, a, rng):
    """回傳 (reward, 下一個 state 或 None)。action 0：S 的 safe / B 的 good。"""
    if s == "S":
        return (1.0, None) if a == 0 else (0.0, "B")
    return (2.0, None) if a == 0 else (-10.0, None)


def a2c_update(theta, v, s, a, p, r, nxt, alpha_critic):
    """A2C 一次更新。advantage 估計 δ = r + γ·V(s') − V(s)；終止時沒有 V(s') 那一項。"""
    v_next = 0.0 if nxt is None else v[nxt]
    delta = r + GAMMA * v_next - v[s]
    v[s] += alpha_critic * delta                    # critic：V(s) 往 target 移動
    theta[s] = actor_step(theta[s], a, p, delta)    # actor：δ 是正的就加強這個 action
    return delta


def run_ext(alpha_critic, seed, episodes=EXT_EPISODES, marks=EXT_MARKS):
    rng = RNG(seed)
    theta = {"S": [0.0, 0.0], "B": [0.0, 0.0]}
    v = {"S": 0.0, "B": 0.0}
    record = {}
    for ep in range(episodes + 1):
        if ep in marks:
            p_short = policy(theta["S"])[1]
            p_good = policy(theta["B"])[0]
            record[ep] = (p_short * p_good, p_good, v["S"], v["B"])
        if ep == episodes:
            break
        s = "S"
        while s is not None:
            p = policy(theta[s])
            a = 1 if rng.u() < p[1] else 0
            r, nxt = mdp_step(s, a, rng)
            a2c_update(theta, v, s, a, p, r, nxt, alpha_critic)
            s = nxt
    return record


def summary(values, stuck_below=0.1):
    mean = sum(values) / len(values)
    spread = max(values) - min(values)
    stuck = sum(1 for x in values if x < stuck_below)
    return f"{mean:6.3f}  {spread:9.3f}  {stuck:>3}/{len(values)}"


def main():
    print("A. 同一段經驗，兩種 advantage（critic 落後的例子）")
    v = {"S": 1.0, "B": -1.0}
    print(f"假設 critic 目前估計 V(S) = {v['S']:g}、V(B) = {v['B']:g}，走了 S→shortcut→B→good")
    d_s = 0.0 + GAMMA * v["B"] - v["S"]
    d_b = 2.0 - v["B"]
    print(f"TD：在 S 的 δ = 0 + {GAMMA}×({v['B']:g}) − {v['S']:g} = {d_s:+.2f}；在 B 的 δ = 2 − ({v['B']:g}) = {d_b:+.2f}")
    g_s = 0.0 + GAMMA * 2.0
    print(f"MC：在 S 的 return = 0 + {GAMMA}×2 = {g_s:.2f}，advantage = {g_s:.2f} − {v['S']:g} = {g_s - v['S']:+.2f}")

    print("\nB. 核心實驗：只改 baseline（兩個 state 的 bandit，數字是 π(R)）")
    for baseline in CORE_BASELINES:
        results = {s: run_core(baseline, s) for s in SEEDS}
        print(f"\nBASELINE = {baseline}")
        print("  episode |  X：平均  最大-最小  卡住 |  Y：平均  最大-最小  卡住")
        for k in CORE_MARKS:
            xs = [results[s][0][k][0] for s in SEEDS]
            ys = [results[s][0][k][1] for s in SEEDS]
            print(f"  {k:>7} | {summary(xs)} | {summary(ys)}")
        final_x = ", ".join(f"{results[s][0][CORE_MARKS[-1]][0]:.3f}" for s in SEEDS)
        final_y = ", ".join(f"{results[s][0][CORE_MARKS[-1]][1]:.3f}" for s in SEEDS)
        print(f"  最後各 seed 的 π(R)：X = [{final_x}]  Y = [{final_y}]")
        if baseline == "shared":
            print("  最後的共用 baseline：" + ", ".join(f"{results[s][2]:.2f}" for s in SEEDS))
        if baseline == "per_state":
            print("  最後的 critic：" + ", ".join(
                f"V(X)={results[s][1]['X']:.2f}/V(Y)={results[s][1]['Y']:.2f}" for s in SEEDS[:3]) + " ……")

    print("\nC. 擴充實驗：A2C，只改 critic 的步長（兩步 MDP，數字是走最佳路線的機率）")
    for alpha_critic in EXT_CRITIC_ALPHAS:
        results = {s: run_ext(alpha_critic, s) for s in SEEDS}
        print(f"\nALPHA_CRITIC = {alpha_critic:g}")
        print("  episode |    平均  最大-最小  停在 safe")
        for k in EXT_MARKS:
            vals = [results[s][k][0] for s in SEEDS]
            print(f"  {k:>7} | {summary(vals)}")
        last = EXT_MARKS[-1]
        print("  最後各 seed：" + ", ".join(
            f"{results[s][last][0]:.3f}(good {results[s][last][1]:.2f}, V(S) {results[s][last][2]:.2f}, V(B) {results[s][last][3]:.2f})"
            for s in SEEDS))

    print("\nD. 看 seed 2、ALPHA_CRITIC = 0.1 的前 300 個 episode")
    trace = run_ext(0.1, 2, episodes=300, marks=(0, 10, 20, 40, 80, 150, 300))
    print("  episode  最佳路線  π(good|B)    V(S)    V(B)")
    for k, (p_opt, p_good, vs, vb) in trace.items():
        print(f"  {k:>7}  {p_opt:8.3f}  {p_good:9.3f}  {vs:6.2f}  {vb:6.2f}")


if __name__ == "__main__":
    main()
