# 第07課：policy gradient——直接學 policy

發布日期：2026-11-05。接續第06課。第01—06課已完成，本課發布不代表完成。

核心問題：之前的 policy 都是先學 q、再從 q 挑 action。如果讓 policy 自己有參數，直接調參數讓平均 return 變大，要怎麼調？只靠試出來的結果，能不能算出調整方向？

## 學習安排與交付

| 環節 | 預計分鐘 | 要完成的事 |
|---|---:|---|
| 閉卷回憶 | 5 | 4 題，來自第06課的待複習 |
| 模組 A 與 B | 30 | A：把 policy 寫成函數、精確梯度。B：只靠抽樣估計梯度（REINFORCE） |
| 休息 | 10 | 離開螢幕 |
| 核心與擴充實驗 | 55 | 核心只改「梯度怎麼來」，擴充只改 OFFSET |
| 綜合小測 | 15 | 5 題集中提交 |
| 學習記錄 | 5 | 結論的適用範圍與剩下的疑問 |

- 運行：執行 [`policy_gradient_bandit.py`](../experiments/policy_gradient_bandit.py)，可對照[參考輸出](../experiments/reference_output.txt)。只用標準庫。亂數產生器是自己寫的，在任何 Python 環境跑出來的數字都一樣。
- 程式已由導師實際執行檢查；導師執行不算作你完成。

## 閉卷回憶（5 分鐘）

1. 會終止的 transition，TD target 是什麼？它和 target network 有沒有關係？
2. 要把一個實驗結論推廣出去之前，第一個該找反例的地方是哪裡？
3. 第06課用 trace、det 判斷收斂，這個判斷有哪兩個前提？
4. 用自己的話說：什麼叫 on-policy？

## 模組 A：把 policy 寫成函數

**第 1 步：之前的 policy 從哪裡來。** 第04課先學 q，再用 ε-greedy 從 q 挑 action。policy 是 q 的附屬品。

**第 2 步：本課的做法。** 讓 policy 自己有參數 θ，直接調 θ，讓平均 return 變大。理由：

- 機器人的 action 是連續的數字（例如關節目標角度），沒辦法對無限多個 action 取最大值。
- θ 小改，機率就小改；argmax 則可能一下子整個換掉。
- Isaac Lab 常用的 PPO（rsl_rl），actor 就是這種「有參數的 policy」。

**第 3 步：最簡單的例子。** 一個 state、兩個 action。這次 L、R 真的是岔路：選一條，拿到 reward，episode 就結束。

```
            ┌─ L：reward 永遠是 1
 state ─────┤
            └─ R：reward 一半機率 4、一半機率 0（平均 2）
```

R 平均比較好，但 agent 一開始不知道。

**第 4 步：softmax 把參數變成機率。** 每個 action 有一個 preference 數字 θ_L、θ_R（教材 9.3 節的 h(s, a, θ)）：

$$
\pi(L)=\frac{e^{\theta_L}}{e^{\theta_L}+e^{\theta_R}},\qquad \pi(R)=\frac{e^{\theta_R}}{e^{\theta_L}+e^{\theta_R}}
$$

| θ = (θ_L, θ_R) | π(R) |
|---|---:|
| (0, 0) | 0.5 |
| (0, 1) | 0.731 |
| (0, 2) | 0.881 |
| (5, 5) | 0.5 |

機率永遠大於 0、加起來是 1，θ 大的那個機率大，而且只有兩個 θ 的差距有影響。

**第 5 步：目標。** 平均 reward：

$$
J(\theta)=\pi(L)\cdot 1+\pi(R)\cdot 2 .
$$

θ = (0, 0) 時 J = 1.5。要讓 J 變大，就沿 gradient 走：θ ← θ + α∇J。第05課是讓 loss 變小，所以減；這裡是讓 J 變大，所以加。

**第 6 步：算 ∇J。** softmax 的導數：∂π(R)/∂θ_R = π(L)π(R)，∂π(R)/∂θ_L = −π(L)π(R)，π(L) 正好相反。代入：

$$
\frac{\partial J}{\partial\theta_R}=\pi(L)\pi(R)\,(2-1),\qquad \frac{\partial J}{\partial\theta_L}=-\pi(L)\pi(R)\,(2-1).
$$

θ = (0, 0) 時 ∇J = (−0.25, +0.25)：R 往上推、L 往下推，力道等於 π(L)π(R) 乘上兩個平均 reward 的差。

一般寫法：∂J/∂θ_a = π(a) × (a 的平均 reward − J)。比整體平均好的 action 被推上去，差的被推下來。

**第 7 步：問題。** 這要知道 R 的平均是 2。真實情況下 agent 不知道，只能試一次、看一次結果。模組 B 說明只靠抽樣也能估計這個梯度。

## 模組 B：只靠抽樣估計梯度（REINFORCE）

**第 1 步：一個小技巧。** ln π 的導數是 ∇π / π，所以

$$
\nabla\pi(a)=\pi(a)\,\nabla\ln\pi(a).
$$

**第 2 步：把 ∇J 寫成平均。** J = Σ π(a)·r̄(a)，r̄(a) 是 a 的平均 reward，和 θ 無關：

$$
\nabla J=\sum_a \nabla\pi(a)\,\bar r(a)=\sum_a \pi(a)\,\big[\nabla\ln\pi(a)\,\bar r(a)\big].
$$

右邊是「按 π 抽一個 a，計算 ∇ln π(a) 乘以 a 的平均 reward」的平均值。這就是教材的式 (9.11)、(9.31)。

**第 3 步：用一次抽樣代替平均。** 和第02課的 MC 一樣：抽一次 a、看到這次的 reward r，用 r·∇ln π(a) 當梯度的一個樣本：

$$
\theta\leftarrow\theta+\alpha\, r\,\nabla\ln\pi(a).
$$

這就是 REINFORCE（教材式 9.32）。

**第 4 步：softmax 的 ∇ln π。** 選到的那格是 1 − π，其他格是 −π。θ = (0, 0) 時：選 L 得 (+0.5, −0.5)；選 R 得 (−0.5, +0.5)。

**第 5 步：一次更新的三種可能**（α = 0.1，實驗 A 段會印出來）：

| 情況 | 機率 | θ 的變化 | 更新後 π(R) |
|---|---:|---|---:|
| 選 L，r = 1 | 0.5 | (+0.05, −0.05) | 0.475 |
| 選 R，r = 4 | 0.25 | (−0.20, +0.20) | 0.599 |
| 選 R，r = 0 | 0.25 | (0, 0) | 0.5 |

按機率平均：0.5×(+0.05, −0.05) + 0.25×(−0.2, +0.2) = (−0.025, +0.025)，正好等於 α∇J。

**第 6 步：選到比較差的 L，也會被加強。** reward 全是正數，所以選到誰就把誰往上推。R 平均被推得比較多，所以平均下來方向仍然對，但每一步都很吵。擴充實驗看這個吵會帶來什麼。

**第 7 步：多步的 episode。** r 換成「從這一步開始的 return」q_t（教材 Algorithm 9.1）。用第04課的兩步 MDP 舉例，γ = 0.9：

```
 S ──shortcut（reward 0）──> B ──good（reward 2）──> 終止
```

- 在 S：q = 0 + 0.9×2 = 1.8，用 1.8·∇ln π(shortcut | S) 更新 S 的 preference。
- 在 B：q = 2，用 2·∇ln π(good | B) 更新 B 的 preference。

要等 episode 結束才算得出 q_t，這和第02課的 MC 一樣。

**第 8 步：為什麼是 on-policy。** 第 2 步的平均是按目前的 π 算的，所以 a 必須照目前的 π 抽，平均才等於 ∇J。用別的 policy 抽 action，就不再是這個梯度（教材 9.4 節）。

**第 9 步：和 PPO 的關係。**【原始碼】rsl_rl 的 PPO 計算 `ratio = exp(log π − log π_old)`，actor loss 是 `−advantages × ratio`（再加上 clip）。θ = θ_old 時，ratio 的梯度正好是 ∇ln π，所以第一步的更新方向是 advantage × ∇ln π：把 REINFORCE 的 q_t 換成 advantage。【推斷】advantage 是 return 減掉 critic 估計的 value，再標準化。為什麼要減，擴充實驗會給線索，第08課正式講。

## 核心實驗（約 30 分鐘）：只改「梯度怎麼來」

設定：θ 從 (0, 0) 開始，α = 0.1，400 步，5 個 seed。表中數字是 π(R)。「精確」那一欄用模組 A 的 ∇J（知道平均 reward），只做對照；REINFORCE 只靠抽樣。

1. **讀 A 段輸出**，對照模組 B 第 5 步。
2. **讀程式**：`exact_gradient` 用到 `MEAN_REWARD`，`run_reinforce` 裡哪一行取代了它？哪一行讓它是 on-policy？
3. **先預測**：第 25 步時，5 個 seed 會不會和精確那一欄一樣？平均呢？
4. **運行並記錄**：第 25、100、400 步的精確值、5 seed 平均、最大−最小。
5. **只改一個條件**：`SEEDS = (6, 7, 8, 9, 10)`。剛才的結論還成立嗎？

## 擴充實驗（約 25 分鐘）：只改 OFFSET

每次拿到的 reward 都加上同一個 OFFSET。L 變成 1 + OFFSET，R 變成 4 + OFFSET 或 OFFSET。R 仍然比 L 好 1。

1. **先想一想**：加上 OFFSET 後，期望的更新會不會變？提示：Σ_a π(a)∇ln π(a) = Σ_a ∇π(a) = ∇(Σ_a π(a)) = ∇1 = 0。
2. **先預測，再運行** `EXT_OFFSETS = (0.0, 10.0)`：記錄最後卡在 L 的 seed 數，以及第 25、100、400 步的最大−最小。
3. **只改一個條件**：`EXT_OFFSETS = (-1.5,)`，也就是每個 reward 減掉 1.5（θ = (0, 0) 時的平均 reward）。和 OFFSET = 0 比，最大−最小有什麼變化？
4. **選做**：`EXT_OFFSETS = (-10.0,)`。

## 綜合小測（5 題，集中提交）

1. **計算**：θ = (0, 1)，選到 L，r = 1，α = 0.1。∇ln π(L) 是多少？更新後的 θ 和 π(R) 是多少？
2. **解釋**：reward 全是正數時，選到比較差的 L，π(L) 也會變大。為什麼整體還是會學到選 R？
3. **程式**：如果把 `a = 1 if rng.u() < p[1] else 0` 改成 `a = 1 if rng.u() < 0.5 else 0`（不管 policy，永遠一半一半），平均的 θ 變化還會等於 α∇J 嗎？可以改一行跑看看。
4. **證據範圍**：OFFSET = 10 時 seed 1—5 有 2 個卡在 L。能不能說「OFFSET = 10 時有 40% 的機率學錯」？為什麼？至少說出一個可以再檢查的條件。
5. **遷移到 PPO**：rsl_rl 的 actor 第一步更新方向是 advantage × ∇ln π。（a）advantage 取代了 REINFORCE 裡的哪個量？（b）advantage = return − value，再減去平均。這最像擴充實驗裡哪個 OFFSET？（c）根據擴充實驗，這樣做的好處是什麼？

## 提交格式

可以分兩批：先交閉卷回憶與模組 A 檢查，實驗與小測稍後一起交。

~~~text
實際用時（未計時可寫未計時）：
結果來源：本地運行 / 分析參考輸出
閉卷回憶 1—4：
核心實驗預測與記錄（第25/100/400步：精確、平均、最大−最小）：
SEEDS 改成 6—10 的結果與判斷：
擴充實驗預測與記錄（OFFSET、卡住的 seed 數、最大−最小）：
OFFSET = −1.5 的結果與解釋：
小測 1—5：
本課結論的適用範圍：
仍不確定的地方：
~~~

## 資料與課程對應

- 趙世鈺《強化學習的數學原理》第 9 章：9.1（policy 從 table 到 function）、9.3（式 9.11 的 ∇π = π∇ln π、式 9.12 的 softmax）、9.4（REINFORCE、式 9.32、Algorithm 9.1、on-policy）。[作者官方倉庫](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)
- Sutton & Barto 第 2 版：2.8 節 Gradient Bandit Algorithms（和本課實驗幾乎同一個設定；圖 2.5 比較 reward 平移時有無 baseline 的差別）、13.3 節 REINFORCE。[線上版](http://incompleteideas.net/book/the-book-2nd.html)
- Williams, 1992, *Simple statistical gradient-following algorithms for connectionist reinforcement learning*, Machine Learning 8。REINFORCE 的原始論文。
- rsl_rl：[`rsl_rl/algorithms/ppo.py`](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py) 的 `ratio` 與 `surrogate`（2026-10-28 版本核對）。

兩個 action 的環境與數字為本課自編的教學設定。
