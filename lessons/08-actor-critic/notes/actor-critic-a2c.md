# 第08課：actor-critic——讓 critic 學出 baseline

發布日期：2026-11-09。接續第07課。第01—07課已完成，本課發布不代表完成。

核心問題：第07課的 baseline 是手動選的 1.5。state 一多、每個 state 平常能拿到的 return 都不同時，baseline 該怎麼選？能不能不用等 episode 結束就更新？

## 學習安排與交付

| 環節 | 預計分鐘 | 要完成的事 |
|---|---:|---|
| 閉卷回憶 | 5 | 4 題，第 1 題是連續兩課答錯的那題 |
| 模組 A 與 B | 30 | A：baseline 要跟著 state 走，由 critic 學出來。B：用 TD 估 advantage（A2C） |
| 休息 | 10 | 離開螢幕 |
| 核心與擴充實驗 | 55 | 核心只改 baseline 的種類；擴充只改 critic 的步長 |
| 綜合小測 | 15 | 5 題集中提交 |
| 學習記錄 | 5 | 結論的適用範圍與剩下的疑問 |

- 運行：執行 [`actor_critic.py`](../experiments/actor_critic.py)，可對照[參考輸出](../experiments/reference_output.txt)。只用標準庫，亂數產生器寫在程式裡。
- 程式已由導師實際執行檢查；導師執行不算作你完成。

## 閉卷回憶（5 分鐘）

1. 會終止的 transition，TD target 是什麼？它和 target network 有沒有關係？
2. REINFORCE 為什麼一定要照目前的 π 抽 action？
3. 每個 reward 都加上同一個數，期望的更新會不會變？哪一點會變差？
4. 第03課：n-step TD 的 n 越大，越接近哪一種方法？

## 模組 A：baseline 要跟著 state 走

**第 1 步：回顧第07課。** 每個 reward 減掉同一個數 b，期望的更新不變（因為 Σ_a π(a)∇ln π(a) = 0），但雜訊會變小。b 選在「平常拿到的 reward」附近效果最好，例如 −1.5。

**第 2 步：兩個 state 的例子。** 每個 episode 隨機出現在 X 或 Y，選一個 action 就結束：

```
 Y：L 永遠 1 分；R 一半機率 4、一半機率 0（平均 2）
 X：和 Y 一樣，但每個 reward 都多 10 分（L 11 分；R 14 或 10 分，平均 12）
```

兩個 state 都是 R 比較好。可是「平常拿到的 reward」差很多：一開始一半一半選時，X 約 11.5，Y 約 1.5。共用一個 b（例如 6.5）放不進兩邊：X 的 reward 全都比 b 大很多，就像第07課的 OFFSET = +5；Y 的全都比 b 小，就像 OFFSET = −5。兩邊都吵。

**第 3 步：每個 state 用自己的 baseline b(s)。** 期望仍然不變：在同一個 state 裡，b(s) 和選哪個 action 無關，所以 Σ_a π(a|s)∇ln π(a|s)·b(s) = b(s)·0 = 0（教材 10.2.1）。

**第 4 步：最自然的 b(s) 是 V(s)。** V(s) 是「從 s 出發，照目前的 policy 平常能拿到的 return」。減掉之後的量叫 advantage：

$$
A(s,a)=q(s,a)-V(s),
$$

意思是「在 s 選 a，比在 s 的平常表現好多少」。正的推上去，負的壓下來。這是第07課一般式 π(a)(r̄(a) − J) 的推廣：J 換成了每個 state 自己的 V(s)。

**第 5 步：V(s) 不知道，就用學的。** 學 V 的方法就是第01—06課的 TD。學 value 的部分叫 **critic**，policy 叫 **actor**：

```
            state s
               │
      ┌────────┴────────┐
  actor π(a|s)      critic V(s)
  選 action          估計「平常有多好」
      │                  │
      ▼                  │
  環境給 r、s'            │
      │                  │
      └──> advantage = 實際結果 − V(s)
             actor：advantage > 0 推上去，< 0 壓下來
             critic：V(s) 往實際結果靠近
```

在兩個 state 的例子裡，每個 episode 只有一步就終止，所以「實際結果」就是 r：TD target 只有 r，沒有 V(s') 那一項。critic 的更新是 V(s) ← V(s) + α_w·(r − V(s))。

## 模組 B：用 TD 估 advantage（A2C）

**第 1 步：多步的 episode。** 「實際結果」有兩種選擇：

- MC：用整個 return G_t。不偏，但要等 episode 結束，而且步數越多越吵。
- TD：用 r + γ·V(s')。每一步就能算，比較不吵，但借用了 critic 對下一個 state 的估計。

**第 2 步：TD 版的 advantage。**

$$
\delta=r+\gamma V(s')-V(s)\qquad\text{（終止時沒有 }V(s')\text{，}\delta=r-V(s)\text{）}
$$

符號注意：教材第10章的 δ 是「target − 估計」，正好和我們之前用的 error（估計 − target）相反，所以下面兩個更新都是加號。

**第 3 步：A2C 每一步做兩個更新**（教材 Algorithm 10.2）：

- critic：V(s) ← V(s) + α_w·δ
- actor：θ_s ← θ_s + α_θ·δ·∇ln π(a|s)

**第 4 步：例子（實驗 A 段會印出來）。** 用第04課的兩步 MDP：

```
 S ──safe（reward 1）──> 終止
 S ──shortcut（reward 0）──> B ──good（reward 2）──> 終止
                             B ──bad（reward −10）──> 終止
```

假設 critic 目前估計 V(S) = 1、V(B) = −1，這次走了 S → shortcut → B → good：

| | 在 S（shortcut） | 在 B（good） |
|---|---|---|
| TD | δ = 0 + 0.9×(−1) − 1 = **−1.9** | δ = 2 − (−1) = +3 |
| MC | 1.8 − 1 = **+0.8** | 2 − (−1) = +3 |

這次 shortcut 明明通往了 good，TD 卻把它往下壓，因為它借用的 V(B) = −1 還停留在「B 常選 bad」的過去。critic 落後時，TD 的 advantage 會被帶偏。

**第 5 步：取捨。** MC 不偏但吵、要等；TD 即時、比較不吵，但 critic 不準時會偏。第03課的 n-step 介於兩者之間。PPO 用的 GAE 用一個 λ 調整：λ = 0 就是 TD，λ = 1 就是 MC。第09課講。

**第 6 步：和 PPO 的關係。**【原始碼】rsl_rl 的 `compute_returns` 裡，每一步先算 `delta = reward + next_is_not_terminal × gamma × next_value − value`，終止時 `next_is_not_terminal` 是 0，正好就是「終止時沒有 V(s') 那一項」。再用 λ（預設 0.95）把 delta 累積成 advantage；returns = advantage + value 是 critic 的 target，value loss 讓 V 靠近它；actor 用標準化後的 advantage。所以 PPO 就是一個 actor-critic。

## 核心實驗（約 30 分鐘）：只改 baseline

設定：兩個 state 的例子，800 個 episode，α_θ = α_w = 0.1，5 個 seed。表中是 π(R)；「卡住」是最後 π(R) < 0.1 的 seed 數。

- `none`：不減 baseline（第07課的 REINFORCE）
- `shared`：所有 state 共用一個 baseline（全部 reward 的移動平均）
- `per_state`：每個 state 一個 V(s)，也就是 critic

1. **讀程式**：`run_core` 裡三種 advantage 分別是哪一行？`per_state` 的 critic 更新，target 為什麼是 r？
2. **先預測**：哪一種的 X 最容易卡住？哪一種會連 Y 也被拖累？
3. **運行並記錄**：第 800 個 episode 各方法 X、Y 的平均與卡住數；`shared` 最後的 b 大約多少？critic 的 V(X)、V(Y) 接近多少？
4. **只改一個條件**：`SEEDS = (6, 7, 8, 9, 10)`。結論還成立嗎？

## 擴充實驗（約 25 分鐘）：A2C，只改 critic 的步長

設定：兩步 MDP，2000 個 episode，5 個 seed。表中是走最佳路線（shortcut → good）的機率；「停在 safe」是最後低於 0.1 的 seed 數。

1. **先想**：`ALPHA_CRITIC = 0` 時 V 永遠是 0。在 S 選 shortcut 的 δ 是多少？選 safe 呢？actor 會學成什麼？
2. **運行並記錄**：三種步長在第 2000 個 episode 的平均與停在 safe 的數目。
3. **讀 D 段**：π(good|B) 在第 10 個 episode 就到 0.99，V(B) 什麼時候才轉正？這段時間裡 shortcut 的 δ 大多是正還是負？最佳路線的機率怎麼變？
4. **只改一個條件**：`SEEDS = (6, 7, 8, 9, 10)`。0.1 和 0.5 的比較還成立嗎？

## 綜合小測（5 題，集中提交）

1. **計算**：V(S) = 1.2、V(B) = 1.5、γ = 0.9，走了 S → shortcut → B → good。在 S 和在 B 的 δ 各是多少？α_w = 0.1 時，V(S) 更新成多少？如果在 B 選的是 bad，δ 是多少？
2. **解釋**：為什麼減掉 V(s) 不會改變期望的更新？為什麼共用一個 b 在 X、Y 都不好？
3. **程式**：`a2c_update` 裡的 `v_next = 0.0 if nxt is None else v[nxt]` 對應公式的哪一部分？如果把它改成永遠用 `v[nxt]`（終止也去查一個 value），會有什麼問題？
4. **證據範圍**：seed 1—5 中，`ALPHA_CRITIC = 0.1` 有 2 個停在 safe，0.5 只有 1 個。能不能說「0.5 比較好」？為什麼？
5. **遷移到 PPO**：（a）rsl_rl 的 `delta` 對應本課哪個量？（b）`next_is_not_terminal` 在終止時是 0，它的作用是什麼？（c）人形機器人一個 episode 常有上千步，為什麼不直接用 MC return，而 PPO 預設 λ = 0.95 介於 TD 與 MC 之間？

## 提交格式

~~~text
實際用時（未計時可寫未計時）：
結果來源：本地運行 / 分析參考輸出
閉卷回憶 1—4：
核心實驗預測與記錄（各方法 X、Y 平均與卡住數；shared 的 b；critic 的 V）：
SEEDS 改成 6—10 的結果與判斷：
擴充實驗預測與記錄（ALPHA_CRITIC、平均、停在 safe 的數目）：
D 段的觀察：
小測 1—5：
本課結論的適用範圍：
仍不確定的地方：
~~~

## 資料與課程對應

- 趙世鈺《強化學習的數學原理》第 10 章：10.1（QAC）、10.2.1（baseline invariance）、10.2.2（advantage、A2C、Algorithm 10.2）。[作者官方倉庫](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)
- Sutton & Barto 第 2 版：13.4 節（REINFORCE with baseline）、13.5 節（actor-critic）。[線上版](http://incompleteideas.net/book/the-book-2nd.html)
- Mnih et al., 2016, *Asynchronous Methods for Deep Reinforcement Learning*（A3C／A2C）。
- Schulman et al., 2016, *High-Dimensional Continuous Control Using Generalized Advantage Estimation*，[arXiv:1506.02438](https://arxiv.org/abs/1506.02438)（GAE，第09課用）。
- rsl_rl：[`rsl_rl/algorithms/ppo.py`](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py) 的 `compute_returns`（2026-10-28 版本核對；預設 γ = 0.99、λ = 0.95）。

兩個 state 的 bandit、兩步 MDP 的數字為本課自編的教學設定。
