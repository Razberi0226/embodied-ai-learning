# 第09課：PPO——同一批資料用好幾次，但不能走太遠

發布日期：2026-11-10。接續第08課。第01—08課已完成，本課發布不代表完成。

核心問題：REINFORCE 和 A2C 的每批資料只用一次就丟掉。收集資料要跑環境，很花時間，能不能同一批多用幾次？多用幾次會出什麼事，PPO 怎麼防？另外，GAE 的 λ 到底在調什麼？

## 學習安排與交付

| 環節 | 預計分鐘 | 要完成的事 |
|---|---:|---|
| 閉卷回憶 | 5 | 4 題，來自第06—08課 |
| 模組 A 與 B | 30 | A：ratio 與 clip。B：GAE 與完整的 PPO 迴圈 |
| 休息 | 10 | 離開螢幕 |
| 核心與擴充實驗 | 55 | 核心只改更新方式（one_step／no_clip／clip）；擴充只改 GAE 的 λ |
| 綜合小測 | 15 | 5 題集中提交 |
| 學習記錄 | 5 | 結論的適用範圍與剩下的疑問 |

- 運行：執行 [`ppo_clip_gae.py`](../experiments/ppo_clip_gae.py)，可對照[參考輸出](../experiments/reference_output.txt)。只用標準庫，亂數產生器寫在程式裡。
- 程式已由導師實際執行檢查；導師執行不算作你完成。

## 閉卷回憶（5 分鐘）

1. δ 是誰減誰？終止時 δ 等於什麼？
2. 第07課第 3 題：不照目前的 π 抽 action、又不修正，結果會怎樣？
3. critic 落後時，TD 的 advantage 會出什麼問題？
4. 第06課的 target network 是什麼想法？PPO 的 critic 哪裡用到了類似的做法？

## 模組 A：ratio 與 clip

**第 1 步：為什麼想重複用資料。** 在 Isaac Lab 裡，收集一批資料要讓幾千個環境各跑幾十步。只做一次梯度就丟掉太浪費。想法：同一批資料做好幾輪更新（epochs）。

**第 2 步：問題。** 第一輪更新完，π 已經變了，但資料是舊的 π 收集的。這就像第07課第 3 題：資料的比例不是目前 policy 的比例，不修正就會學偏。

**第 3 步：修正——ratio。**

$$
\text{ratio}=\frac{\pi_{\text{新}}(a|s)}{\pi_{\text{舊}}(a|s)}
$$

每筆資料的推力乘上 ratio：新 policy 比較常選這個 action，就算多一點；比較少選，就算少一點。actor 的目標變成「平均 ratio × advantage」。剛開始 θ = θ_舊，ratio = 1，梯度就是 advantage × ∇ln π，和第07課第 9 步一樣。

**第 4 步：新問題——ratio 可以變得很大。** 實驗 B 段的例子（seed 6 第 18 輪，不 clip）：

- 收集時 π(R) = 0.994、π(L) = 0.006，已經學得很好。
- 這批 4 筆資料是 R 0 分、L 1 分、R 0 分、R 0 分。批平均 0.25，所以 L 的 advantage 是 +0.75，三筆 R 都是 −0.25。
- 10 個 epoch 裡，L 一直被推上去：ratio(L) 從 1.2 變成 163。
- 結果 π(R) 從 0.994 掉到 0.0016。之後每批幾乎都只有 L、reward 都是 1、advantage 全是 0，再也學不回來。

一批只有 4 筆資料的運氣，就把學好的 policy 毀了。

**第 5 步：clip。** 只讓 ratio 在 [1 − ε, 1 + ε] 之內起作用（ε = 0.2）：

| 情況 | 做法 |
|---|---|
| advantage > 0，ratio 已經 > 1.2 | 這筆已經推夠了，不再推 |
| advantage < 0，ratio 已經 < 0.8 | 這筆已經壓夠了，不再壓 |
| 其他 | 照常更新 |

寫成公式（Schulman et al., 2017）：

$$
L^{\text{CLIP}}=\text{平均}\Big[\min\big(\text{ratio}\cdot A,\ \text{clip}(\text{ratio},1-\varepsilon,1+\varepsilon)\cdot A\big)\Big].
$$

取 min 的效果就是上表：超過範圍、而且還想往同方向推時，用的是被截住的那一項，它的梯度是 0。同一個例子加上 clip，ratio(L) 停在 1.2，π(R) 只從 0.9939 變成 0.9925。

**第 6 步：rsl_rl 怎麼寫。**【原始碼】`surrogate = −advantage × ratio`、`surrogate_clipped = −advantage × clamp(ratio, 1 − clip, 1 + clip)`，loss 取兩者較大的那個（等於上面的 −min）。預設 `clip_param = 0.2`、`num_learning_epochs = 5`、`num_mini_batches = 4`。另外還有第二道保險：`schedule = "adaptive"` 時，每次更新都量新舊 policy 的差距（KL）。KL 超過 `desired_kl`（預設 0.01）的 2 倍，learning rate 就除以 1.5；低於一半就乘以 1.5。兩道保險都是為了不讓 policy 一次走太遠。

## 模組 B：GAE 與完整的 PPO 迴圈

**第 1 步：回顧第08課。** TD 的 δ 偏但穩；MC 不偏但吵；第03課的 n-step 介於兩者之間。

**第 2 步：GAE。** 把後面每一步的 δ 打折加起來：

$$
A_t=\delta_t+\gamma\lambda\,\delta_{t+1}+(\gamma\lambda)^2\,\delta_{t+2}+\cdots
$$

從最後一步往回算更方便：A_t = δ_t + γλ·A_{t+1}，終止時後面那一項是 0。這正是 rsl_rl `compute_returns` 裡 `advantage = delta + next_is_not_terminal × gamma × lam × advantage` 那一行。

- λ = 0：A_t = δ_t，就是第08課的 A2C。
- λ = 1：中間的 V 會兩兩抵消，A_t = G_t − V(s_t)，就是 MC 加 baseline。

**第 3 步：小例子。** γ = 0.9、λ = 0.5，兩步的 episode，δ_0 = 1、δ_1 = 2（第 1 步終止）：A_1 = 2；A_0 = 1 + 0.9×0.5×2 = 1.9。若 λ = 0，A_0 = 1；若 λ = 1，A_0 = 1 + 0.9×2 = 2.8。

**第 4 步：λ 的取捨。** critic 不準時，λ 小就偏；λ 大就吵。最好的 λ 取決於 critic 有多準、雜訊多大、episode 多長。擴充實驗會量出來。

**第 5 步：完整的 PPO 迴圈。**

```
 ┌─> 1. 用目前的 policy 在所有環境各跑固定步數，存下 (s, a, r, 是否終止, log π_舊, V_舊)
 │   2. 從後往前算 δ 與 GAE advantage；returns = advantage + V_舊；advantage 標準化
 │   3. 重複 epochs × mini-batches 次：
 │        actor：用 clip 的目標（第 A 模組）
 │        critic：讓 V 靠近 returns（returns 在這一輪固定，像第06課的 target network）
 │        再加上 entropy 獎勵，讓 policy 別太早變成只選一個 action
 └── 4. 丟掉這批資料，回到 1（所以 PPO 仍是 on-policy：每批資料只在這一輪用）
```

**第 6 步：對照 Isaac Lab 的設定。** Isaac Lab 的 rsl_rl 設定裡，`num_steps_per_env` 是每個環境每輪跑幾步；`algorithm` 底下的 `clip_param`、`num_learning_epochs`、`num_mini_batches`、`gamma`、`lam`、`desired_kl`、`entropy_coef`、`value_loss_coef` 都對應上面的步驟。

## 核心實驗（約 30 分鐘）：只改更新方式

設定：第07課的 bandit（L 永遠 1 分；R 一半 4 分、一半 0 分）。每輪只收集 4 筆資料，advantage = reward − 這批的平均；共 30 輪；10 個 seed。

- `one_step`：每批只用一次（第07、08課的做法）
- `no_clip`：每批用 10 次，用 ratio 修正，但不 clip
- `clip`：每批用 10 次，用 ratio 修正並 clip（PPO）

1. **讀程式**：`update` 裡哪一行算 ratio？哪幾行做 clip？`one_step` 只用一次，為什麼不用管 ratio？
2. **先預測**：哪一種前期最快？哪一種可能崩壞？
3. **運行並記錄**：第 10、30 輪的平均與最小值；每個 seed 單輪最大的下降；最後崩壞的數目。
4. **讀 B 段**：為什麼 no_clip 的 ratio(L) 能長到 163？clip 為什麼停在 1.2？
5. **只改一個條件**：`CLIP_EPS = 0.5`。先預測 clip 那一組會怎樣，再運行。

## 擴充實驗（約 25 分鐘）：只改 GAE 的 λ

設定：s0 選 R 或 L（policy 固定一半一半，只比較梯度估得準不準，不訓練）。選完各自再走 30 步：R 這條每步平均 0.1 分，L 這條平均 0 分，每步都有 ±1 的雜訊。critic 不完美（品質 0.5：只學到兩條路差別的一半）。每次用 16 個 episode 估一次梯度，重複 300 次，看估計的平均、偏差、分散和總誤差（RMSE）。

1. **先預測**：λ 從 0 到 1，偏差和分散各怎麼變？
2. **運行並記錄**：C 段的表格；哪個 λ 的總誤差最小？
3. **只改一個條件**：`CRITIC_QUALITY = 1.0`，再試 `0.0`。最好的 λ 各變成多少？為什麼？
4. **證據範圍**：能不能說「λ = 0.95 永遠最好」？

## 綜合小測（5 題，集中提交）

1. **計算（clip）**：ε = 0.2。下面四種情況，哪幾種還會繼續推（梯度不是 0）？min(ratio·A, clip(ratio)·A) 各是多少？（a）A = +2、ratio = 1.1；（b）A = +2、ratio = 1.5；（c）A = −1、ratio = 0.9；（d）A = −1、ratio = 0.6。
2. **計算（GAE）**：γ = 0.9、λ = 0.8，三步的 episode，δ_0 = 0.5、δ_1 = −1、δ_2 = 2（第 2 步終止）。求 A_2、A_1、A_0。
3. **解釋**：B 段裡 no_clip 崩壞之後，為什麼再也學不回來？PPO 的 entropy 獎勵對這種情況有什麼幫助？
4. **證據範圍**：no_clip 在 10 個 seed 裡崩壞 1 個。能不能說「不 clip 有 10% 的機率崩壞」？如果把 `BATCH` 改成 16，你預期會怎樣？這說明了什麼？
5. **遷移到 Isaac Lab**：（a）rsl_rl 設定裡哪些參數決定「同一批資料用幾次」？（b）哪些參數在防止「一次走太遠」？（c）如果你看到 reward 原本漲得很好、突然大幅掉下來，根據這一課，你會先檢查哪些設定或曲線？

## 提交格式

~~~text
實際用時（未計時可寫未計時）：
結果來源：瀏覽器 / 電腦本地 / 分析導師輸出
閉卷回憶 1—4：
核心實驗預測與記錄（第10/30輪平均與最小值、單輪最大下降、崩壞數）：
B 段的解釋：
CLIP_EPS = 0.5 的預測與結果：
擴充實驗預測與記錄（λ、偏差、分散、RMSE）：
CRITIC_QUALITY 改成 1.0、0.0 的結果與解釋：
小測 1—5：
本課結論的適用範圍：
仍不確定的地方：
~~~

## 課堂補充（上課時的追問與重講）

以下是上課時針對追問補充的說明。rsl_rl 的行號以 2026-10-28 的版本為準。

### 補充 1：batch、mini-batch、epoch

| 詞 | 意思 |
|---|---|
| batch | 這一輪收集到的全部資料 |
| mini-batch | 從 batch 切出來的一小份；一次更新只用一份，份內的推力取平均 |
| epoch | 全部資料用過一遍 |

以 4096 個環境 × 每個 24 步為例，一輪收 98,304 筆：

| num_mini_batches | 每份筆數 | 每輪更新次數（5 個 epoch） | 等於 |
|---|---|---|---|
| 1 | 98,304 | 5 | 整批更新（第05課的 batch gradient） |
| 4（預設） | 24,576 | 20 | 中間 |
| 98,304 | 1 | — | 一筆一筆更新 |

- 本課程式的 `BATCH = 4` 是「每輪收幾筆」，不切（份數 = 1）；`/ BATCH` 是把 4 筆的推力取平均。rsl_rl 也取平均（`ppo.py` 第 274 行 `.mean()`），這樣步長和筆數無關。
- 各環境不同步沒關係：每筆資料是獨立的一行 (s, a, π_舊(a|s), A)。唯一要按時間順序算的 advantage，是在打散之前由每個環境沿自己的時間往回算好的（`ppo.py` 第 176—184 行，每個環境用自己的 `dones` 切斷）。
- 算完後攤平、整批打亂（`rollout_storage.py` 第 232 行 `torch.randperm(batch_size)`），再切份，所以每一份都混有各步數、各環境的資料。這個版本每輪只打亂一次，5 個 epoch 用同一種切法。用 RNN 的 policy 例外，按整段軌跡切。
- 份數 × epoch = 一輪更新幾次 = policy 能離 π_舊 走多遠。mini-batch 不是越大越好：每份越大，每步梯度越準，但每輪更新次數越少；份數越多，走得越遠，越需要 clip。算力夠時通常是開更多環境，而不是把同一批切更少份。

### 補充 2：clip 之後推力為什麼是 0

**目標值和推力是兩回事。** 推力是目標值對 θ 的斜率。θ 透過 ratio 影響目標，所以推力是兩層斜率相乘：

$$
\text{推力}=\underbrace{(\text{目標對 ratio 的斜率})}_{\text{隱藏因子}}\times\underbrace{(\text{ratio 對 }\theta\text{ 的斜率})}_{=\ \text{ratio}\times\nabla\ln\pi}
$$

| | 隱藏因子 | ratio 對 θ 的斜率 | 推力 |
|---|---|---|---|
| 範圍內（目標 = ratio × A） | A × 1 | ratio × ∇ln π | A × ratio × ∇ln π |
| 範圍外且被 min 選中（目標 = 1.2 × A，常數） | A × 0 | ratio × ∇ln π（不是 0） | 0 |

B 段第 2 次更新、L 這一行（A = +0.75、ratio = 1.2047、∇ln π 的 θ_L 分量 0.9926）：no_clip 推力 0.897；clip 推力 0；如果誤以為「推力最多 1.2 倍」，推力是 0.893，幾乎沒擋。

**限速和圍欄。** 導師另外把「推力倍數最多 1.2」寫進程式測試（不在本課腳本裡）：

| B 段第幾次更新後 | 推力倍數上限 1.2：ratio(L) | π(R) | PPO clip：ratio(L) | π(R) |
|---|---|---|---|---|
| 1 | 1.21 | 0.9926 | 1.21 | 0.9926 |
| 5 | 2.93 | 0.9821 | 1.21 | 0.9926 |
| 10 | 8.64 | 0.9472 | 1.22 | 0.9925 |

10 個 seed、30 輪：推力倍數上限的做法 0/10 崩壞、第 30 輪平均 0.998，但單輪最大下降 −0.41；clip 單輪最大下降 −0.17。PPO 要保證的是每一輪走不遠，不是學得最快。

**判斷三步：**

1. 甲 = ratio × A（沒截），乙 = clip(ratio, 1 − ε, 1 + ε) × A（截過）。
2. 目標值 = min(甲, 乙)。負數要小心：−2.6 < −2.4。
3. 選到甲 → 推；選到乙而且 ratio 在範圍外 → 乙是常數，不推。

min 只擋「往有利方向走過頭」：

| A | ratio | 結果 |
|---|---|---|
| > 0 | ≤ 1 + ε | 推 |
| > 0 | > 1 + ε | 不推 |
| < 0 | ≥ 1 − ε | 壓 |
| < 0 | < 1 − ε | 不壓 |

A > 0 而 ratio 已經小於 1 − ε（好 action 反而變少），或 A < 0 而 ratio 大於 1 + ε（壞 action 反而變多），都照推，讓它修回來。

### 補充 3：∇J 與 PPO 目標的關係

第07課：∂J/∂θ_a = π(a)(r̄(a) − J)。不知道 r̄ 時用 ∇π = π∇ln π 改寫成「照 π 抽 a，取 r × ∇ln π(a) 的平均」；softmax 的 ∇ln π(a) = e_a − π（程式的 `grad_log`）；r 換成 advantage。資料來自 π_舊 時再乘 ratio。而 ratio × A 對 θ 的斜率正好是 ratio × A × ∇ln π，所以 PPO 的目標就是「斜率等於 ∇J 估計」的函數，clip 在走過頭的地方把它截平。

### 補充 4：GAE 權重與 critic 品質

rsl_rl 預設 γλ = 0.99 × 0.95 = 0.9405，往後第 k 步的 δ 的權重：

| k | 1 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|
| 0.9405^k | 0.94 | 0.74 | 0.54 | 0.29 | 0.05 |

擴充實驗的 critic：估計 = 兩條路的平均 + 品質 × (真值 − 平均)。R 路第 1 步真值 2.60、L 路 0、平均 1.30；品質 0.5 時 R 路估 1.95。λ = 0 時選 R 的 advantage ≈ 0.99 × 1.95 − 1.29 ≈ 0.64，只有真實 1.29 的一半，所以梯度估計 0.323 也約是真實 0.644 的一半。λ = 1 時中間的 V 全部抵消，那一列在三種品質下完全相同；其偏差 −0.019 是 300 次平均的抽樣誤差（約 0.340 / √300 ≈ 0.02）。

## 資料與課程對應

- Schulman et al., 2017, *Proximal Policy Optimization Algorithms*，[arXiv:1707.06347](https://arxiv.org/abs/1707.06347)。
- Schulman et al., 2016, *High-Dimensional Continuous Control Using Generalized Advantage Estimation*，[arXiv:1506.02438](https://arxiv.org/abs/1506.02438)。
- 趙世鈺《強化學習的數學原理》第 10 章 10.3.1（importance sampling，ratio 的由來）。PPO 本身不在教材內。[作者官方倉庫](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)
- rsl_rl：[`rsl_rl/algorithms/ppo.py`](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py) 的預設參數、`compute_returns`、clip 與 adaptive learning rate（2026-10-28 版本核對）。

bandit 與兩條路的環境、數字為本課自編的教學設定。
