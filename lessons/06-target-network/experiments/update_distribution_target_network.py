"""第06課：critic 怎樣學得穩——更新分佈與 target network。
只用 Python 標準庫；更新次數固定，並有數值 guard，不會死循環。
環境是教學自編的兩個 state：真實 value 已知，所以能直接算真實誤差。
"""

GAMMA = 0.9
ALPHA = 0.1
START = (1.0, 0.0)            # (w1, w0)，和預習相同；v_hat(x) = w1*x + w0

# 兩種 transition（同一個 policy 的一個 episode 會各經過一次）：
#   T1: x=1 -> x=2，r=0，未終止
#   T2: x=2 -> 終止，r=1
T1 = {"name": "T1", "x": 1.0, "r": 0.0, "x_next": 2.0, "terminal": False}
T2 = {"name": "T2", "x": 2.0, "r": 1.0, "x_next": None, "terminal": True}
TRUE_V = {1.0: 0.9, 2.0: 1.0}  # v_pi(1)=0.9、v_pi(2)=1；本例的直線能精確表示（w1=0.1, w0=0.8）

# 核心實驗：只改 P_T1 = 每 100 次更新中有幾次是 T1。on-policy 比例是 50。
CORE_P_T1 = (50, 80, 95)
CORE_UPDATES = 600
CORE_MARKS = (0, 50, 100, 200, 400, 600)

# 擴充實驗：P_T1 固定 95，只改 SYNC_EVERY（target network 每幾次更新同步一次）。
# SYNC_EVERY = 1 等於沒有 target network。
EXT_P_T1 = 95
EXT_SYNC = (1, 20, 100)
EXT_UPDATES = 1000
EXT_MARKS = (0, 100, 200, 400, 600, 800, 1000)

GUARD = 1e6


def v_hat(x, w):
    w1, w0 = w
    return w1 * x + w0


def pick(k, p_t1):
    """第 k 次更新用哪個 transition：固定週期、平均分散，不用隨機數。"""
    use_t1 = -(-k * p_t1 // 100) > -(-(k - 1) * p_t1 // 100)
    return T1 if use_t1 else T2


def expected_matrix(p_t1):
    """平均更新 w <- w + alpha*(b - A w) 裡的 A = E[phi (phi - gamma*phi_next)^T]。"""
    p = p_t1 / 100
    rows = [[0.0, 0.0], [0.0, 0.0]]
    for t, weight in ((T1, p), (T2, 1 - p)):
        phi = (t["x"], 1.0)
        nxt = (0.0, 0.0) if t["terminal"] else (t["x_next"], 1.0)
        d = (phi[0] - GAMMA * nxt[0], phi[1] - GAMMA * nxt[1])
        for i in range(2):
            for j in range(2):
                rows[i][j] += weight * phi[i] * d[j]
    return rows


def true_error(w):
    return max(abs(v_hat(x, w) - v) for x, v in TRUE_V.items())


def train(p_t1, updates, sync_every, marks):
    w = START
    w_target = START                      # target network 的 parameters
    history = []
    window = []                           # 兩個記錄點之間的 0.5*error^2
    status = "budget_complete"
    history.append((0, w, None, true_error(w)))
    for k in range(1, updates + 1):
        t = pick(k, p_t1)
        # 1) TD target：用 target network 的 w_target 算，這一步當常數
        if t["terminal"]:
            target = t["r"]
        else:
            target = t["r"] + GAMMA * v_hat(t["x_next"], w_target)
        # 2) error = prediction - target；3) gradient = (error*x, error)
        error = v_hat(t["x"], w) - target
        w1, w0 = w
        w = (w1 - ALPHA * error * t["x"], w0 - ALPHA * error)
        window.append(0.5 * error * error)
        # 4) 每 sync_every 次更新，把 main 的 w 複製給 target network
        if k % sync_every == 0:
            w_target = w
        if not all(abs(v) <= GUARD for v in w):
            status = "parameter_guard_stop"
            history.append((k, w, sum(window) / len(window), true_error(w)))
            break
        if k in marks:
            history.append((k, w, sum(window) / len(window), true_error(w)))
            window = []
    return history, status


def show(label, p_t1, updates, sync_every, marks):
    history, status = train(p_t1, updates, sync_every, marks)
    print(f"\n{label}  P_T1={p_t1}  SYNC_EVERY={sync_every}  ALPHA={ALPHA}")
    print(" update        w1          w0    v_hat(1)    v_hat(2)     td_loss  true_error")
    for k, w, loss, err in history:
        loss_text = "         -" if loss is None else f"{loss:>10.4g}"
        print(f"{k:>7} {w[0]:>9.4g} {w[1]:>11.4g} {v_hat(1, w):>11.4g} {v_hat(2, w):>11.4g}"
              f"  {loss_text}  {err:>10.4g}")
    used = history[-1][0]
    print(f"updates_used={used}  status={status}")


def main():
    print("A. 平均更新的矩陣 A（只看 trace 和 det）")
    for p_t1 in CORE_P_T1:
        a = expected_matrix(p_t1)
        trace = a[0][0] + a[1][1]
        det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
        print(f"P_T1={p_t1:>3}  A=[[{a[0][0]:.3f}, {a[0][1]:.3f}], [{a[1][0]:.3f}, {a[1][1]:.3f}]]"
              f"  trace={trace:.3f}  det={det:.4f}")

    print("\nB. 核心實驗：只改更新分佈 P_T1（沒有 target network）")
    for p_t1 in CORE_P_T1:
        show("core", p_t1, CORE_UPDATES, 1, CORE_MARKS)

    print("\nC. 擴充實驗：P_T1 固定，只改 target network 的 SYNC_EVERY")
    for sync in EXT_SYNC:
        show("ext", EXT_P_T1, EXT_UPDATES, sync, EXT_MARKS)


if __name__ == "__main__":
    main()
