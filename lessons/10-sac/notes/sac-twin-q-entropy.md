# 第10課：SAC——把舊資料存起來一直用，再把「保持隨機」寫進目標

發布日期：2026-11-11。接續第09課。第01—09課已完成，本課發布不代表完成。

核心問題：PPO 的每批資料只在一輪裡用，之後就丟掉。資料很貴的時候（例如真機器人），能不能把舊資料存起來一直用？為什麼 SAC 用舊資料不需要 ratio？Q 為什麼會越學越偏高，兩個 Q 怎麼把它壓下來？還有，α 這個「隨機程度」的旋鈕到底在調什麼？

## 學習安排與交付

| 環節 | 預計分鐘 | 要完成的事 |
|---|---:|---|
| 閉卷回憶 | 5 | 4 題，來自第04、06、09課 |
| 模組 A 與 B | 30 | A：replay buffer、為什麼不用 ratio、高估與 twin Q、soft target。B：entropy 與 α、SAC 的 policy、自動調 α、完整迴圈 |
| 休息 | 10 | 離開螢幕 |
| 核心與擴充實驗 | 55 | 核心只改 Q 的數量（single／twin）；擴充只改 α |
| 綜合小測 | 15 | 5 題集中提交 |
| 學習記錄 | 5 | 結論的適用範圍與剩下的疑問 |

- 運行：執行 [`sac_twin_q_alpha.py`](../experiments/sac_twin_q_alpha.py)，可對照[參考輸出](../experiments/reference_output.txt)。只用標準庫，亂數產生器寫在程式裡。
- 程式已由導師實際執行檢查；導師執行不算作你完成。
- 為了看清楚原理，實驗用表格（每個 state、action 一格）代替神經網路，policy 直接取 SAC 的 actor 想學到的樣子。

## 閉卷回憶（5 分鐘）

1. （第09課）ε = 0.2，A = +2，ratio = 1.4。用三步法寫出甲、乙、目標值，這一筆推不推？
2. （第09課）rsl_rl 一輪裡「同一批資料用幾次」由哪兩個參數決定？為什麼 PPO 每輪結束要把資料丟掉？
3. （第04課）Q-learning 為什麼可以用別的 policy 收集的資料來學（off-policy）？
4. （第06課）target network 是什麼？hard update 和 soft update 差在哪？

## 模組 A：用舊資料學 Q——replay buffer、twin Q、soft target

**第 1 步：從 PPO 的限制出發。** PPO 每批資料在一輪裡用 5 次（`num_learning_epochs`），然後丟掉。原因是 actor 的推力 ratio × A × ∇ln π 要求資料來自「接近目前」的 π；離太遠，ratio 就不可靠（第09課 B 段）。在 Isaac Lab 開 4096 個環境，資料很便宜，丟掉沒關係；真機器人一小時只能收幾千步，丟掉就很浪費。

**第 2 步：replay buffer。** 每走一步，把 (s, a, r, s', 是否結束) 存進一個大資料庫，叫 replay buffer。每次更新從裡面隨機抽一小批（mini-batch）。buffer 滿了，就把最舊的擠掉。

- 一筆資料平均被用幾次 ≈ 每次抽幾筆 × 每走一步更新幾次（buffer 穩定以後）。
- Stable-Baselines3 的 SAC 預設每次抽 256 筆、每走一步更新 1 次 → 每筆資料平均被用約 256 次。PPO 是 5 次。
- 本課實驗：每走一步，每個 Q 抽 4 筆 → 每個 Q 平均把每筆資料用 4 次。
- 隨機抽，而不是照順序用，可以打散前後步驟之間的相關性（趙書 8.4 節 DQN 的 experience replay）。

**第 3 步：為什麼用舊資料不需要 ratio。** SAC 的 critic 學的是 Q(s, a)：「在 s 做了 a 之後，值多少」。一筆資料 (s, a, r, s') 說的是：在 s 做了 a，拿到 r，到了 s'。

- a 是「條件」，不是要猜的東西；在 s 做了 a 之後，r 和 s' 由環境決定，和當初是誰選了 a 無關。
- 只有「到了 s' 之後怎麼走」和 policy 有關，這部分用**目前的** π 重新算。

| | 資料裡哪一部分和「誰收集的」有關 | 要不要修正 |
|---|---|---|
| PPO 的 actor：平均 [A × ∇ln π] | 每個 action 出現的頻率（照 π_舊 抽的） | 要，乘 ratio |
| SAC 的 critic：Q(s, a) 的 target | 沒有：a 是給定的，r、s' 由環境決定，s' 之後用目前的 π 算 | 不用 |

這和第04課 Q-learning 能 off-policy 是同一個道理。但舊資料還是有代價：buffer 裡哪些 (s, a) 比較多，是由以前的 policy 決定的，也就是第06課講的「更新分佈」。off-policy＋bootstrap＋function approximation 三個一起出現（deadly triad）時，critic 容易不穩。target network 和下面的 twin Q 都是為了讓它穩一點。

**第 4 步：SAC 的 target 長什麼樣。** 先把符號說清楚：

- Q̄：target network 版本的 Q（第06課），算 target 時用它。
- π(a'|s')：目前的 policy 在 s' 選 a' 的機率。
- α：溫度，模組 B 會講；現在只要知道 α 很小時，下面的 V(s') 幾乎等於「π 最想選的那個 action 的 Q̄」，也就是接近 max。

$$
y=r+\gamma\,(1-\text{結束})\;V(s'),\qquad V(s')=\sum_{a'}\pi(a'|s')\Big[\bar Q(s',a')-\alpha\ln\pi(a'|s')\Big]
$$

然後 Q(s, a) ← Q(s, a) + LR × (y − Q(s, a))，和第04課的 Q-learning 一樣是「往 target 走一步」。

**第 5 步：問題——Q 會偏高（maximization bias）。** 用實驗 B 段的數字：B 有 8 個 action，真值都是 −0.1。每個 Q 的估計是 5 筆 reward 的平均，有雜訊。

- 第 1 次抽樣的 Q1 = −0.07, −0.15, +0.09, −0.36, +0.20, +0.03, +0.29, +0.65。
- 單看每個 action，估計是不偏的：重複 1000 次，8 個 action 的平均是 −0.104，很接近 −0.1。
- 但 V(B) 要取「最好的那個」。最好的那個，往往是「雜訊剛好偏高」的那個。這一次 max = +0.65；重複 1000 次的平均是 +0.451，比真值高了 0.55。

這個偏高的 V(B) 會被拿去算前一步的 target：Q(A, right) = 0 + 0.99 × V(B)。B 明明比「直接結束（0 分）」差，卻看起來比較好，agent 就常常往 B 走。而且 target 會一路往前傳（bootstrap），高估也跟著往前傳。（Sutton & Barto 6.7 節。）

**第 6 步：twin Q（clipped double Q）。** 同時學兩個 Q：Q1、Q2。算 target 時，每個 action 先取兩者中較小的那個，再算 V：

$$
y=r+\gamma\,(1-\text{結束})\sum_{a'}\pi(a'|s')\Big[\min\big(\bar Q_1,\bar Q_2\big)(s',a')-\alpha\ln\pi(a'|s')\Big]
$$

- Q1、Q2 都往**同一個** y 學。
- 同一個例子：Q2 = +0.16, −0.25, −0.57, −0.09, −0.06, +0.16, −0.47, +0.33；min(Q1, Q2) = −0.07, −0.25, −0.57, −0.36, −0.06, +0.03, −0.47, +0.33，最大值從 +0.65 降到 +0.33。重複 1000 次的平均從 +0.451 降到 +0.130。
- 為什麼有效：要兩個 Q 剛好在**同一個** action 上都偏高，機率小很多。
- 代價：min 本身偏低，所以 twin 不保證剛好不偏，有時會變成低估。（+0.130 仍高於 −0.1，說明這裡還沒壓完。）
- 表格版本裡，兩個 Q 如果吃一模一樣的資料就會一模一樣，min 沒有作用。真實 SAC 的兩個網路初始值不同、近似誤差也不同；實驗裡用「兩個 Q 各從 buffer 抽自己的一批」模擬這種差異。

**第 7 步：soft target update。** target network 不是每隔 C 步整個複製（DQN 的 hard update，C = 10000），而是每一步都往主網路靠一點：

$$
\bar Q\leftarrow\bar Q+\tau\,(Q-\bar Q)
$$

- SAC 預設 τ = 0.005：每步只靠近 0.5%。舊值剩一半大約要 138 步（0.995^138 ≈ 0.5）。
- 本課實驗的問題很小，用 τ = 0.05，舊值剩一半大約要 14 步。

**模組 A 檢查**

5. 一筆資料 (s = A, a = right, r = 0, s' = B)，γ = 0.9。B 有兩個 action：Q̄1(B, ·) = (0.4, −0.2)，Q̄2(B, ·) = (0.1, 0.3)。α 很小（V(B) ≈ 取 min 之後的最大值）。求 min(Q̄1, Q̄2)(B, ·)、V(B)、twin 的 target y；只用 Q̄1 時 y 又是多少？
6. 用一句話說：為什麼 SAC 的 critic 用很久以前存的資料也不需要 ratio，PPO 的 actor 卻需要？

## 模組 B：maximum entropy——α 在調什麼

**第 1 步：回顧 entropy。** entropy = −Σ_a π(a) ln π(a)，衡量 policy 有多分散。兩個 action 時：

| π | (0.5, 0.5) | (0.9, 0.1) | (0.998, 0.002) |
|---|---|---|---|
| entropy | 0.693（最大） | 0.325 | 0.014 |

第09課 B 段崩壞之後 π(L) ≈ 0.998，entropy 只剩 0.014，policy 幾乎不再嘗試別的 action。

**第 2 步：SAC 的目標。** 每一步除了 reward，「還保持多少隨機」也算分：

$$
J=\text{平均}\Big[\sum_t\gamma^t\big(r_t+\alpha\cdot\text{entropy}(\pi(\cdot|s_t))\big)\Big]
$$

α（temperature，溫度）決定「1 單位的 entropy 值多少 reward」。

**第 3 步：有了 entropy 之後，最好的 policy 長什麼樣。**

$$
\pi(a|s)\propto\exp\big(Q(s,a)/\alpha\big)
$$

這就是第07課的 softmax，只是把 θ 換成 Q/α。兩個 action、Q = (1, 0) 時，選第二個（比較差的）的機率：

| α | 0.05 | 0.3 | 1 | 3 |
|---|---|---|---|---|
| π(比較差的) | 0.000000002 | 0.034 | 0.269 | 0.417 |

α 小：幾乎只選 Q 最大的（接近 greedy）。α 大：接近平均分配。

（想看推導可以看這段：要讓 Σ_a π(a)Q(a) + α·entropy 最大、而且 Σπ = 1。對 π(a) 微分得 Q(a) − α(ln π(a) + 1) − λ = 0，所以 ln π(a) = Q(a)/α − 常數，也就是 π ∝ exp(Q/α)。）

**第 4 步：entropy 怎麼進到 Q。** 模組 A 的 V(s) = Σ π(a|s)[Q(s,a) − α ln π(a|s)]，拆開就是：

$$
V(s)=\underbrace{\sum_a\pi(a|s)Q(s,a)}_{\text{平均 Q}}+\alpha\cdot\underbrace{\Big(-\sum_a\pi(a|s)\ln\pi(a|s)\Big)}_{\text{entropy}}
$$

例子：B 的 8 個 action 真值都是 −0.1，π 平均分配，entropy = ln 8 = 2.079。α = 0.01 時，V(B) = −0.1 + 0.01 × 2.079 = −0.079。這就是核心實驗印出來的「真值 V(B) = −0.079」。

**第 5 步：α 太小、太大各會怎樣。**

- 太小：幾乎不探索。只要某條路一開始看起來比較好，就再也不試別的路，可能永遠不知道別的路其實更好。
- 太大：太隨機，就算知道哪條路好，也常常走錯；而且 entropy 的分數會蓋過 reward。
- α 要和 reward 的大小一起看：policy 只看 Q/α。reward 全部乘 10，Q 也乘 10，α 要跟著乘 10 才一樣。

**第 6 步：自動調 α（SAC 第二版）。** 先設一個目標 entropy H̄。每一步看目前的 entropy：

- 比 H̄ 低（太不隨機）→ α 變大，鼓勵多探索。
- 比 H̄ 高（太隨機）→ α 變小。

寫成更新式：ln α ← ln α − lr × (目前的 entropy − H̄)。Stable-Baselines3 的程式碼是 `ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()`，其中 −log_prob 的平均就是 entropy；`target_entropy='auto'` 時 H̄ = −(動作維度)（連續動作的 entropy 可以是負的），`ent_coef='auto'` 時 α 的起始值是 1.0。

自動調整保證的是「entropy 到達目標值」，不保證探索一定夠；目標 entropy 還是要人選。好處是它不受 reward 大小影響。

**第 7 步：actor 怎麼學（簡介）。** SAC 的 actor 要讓 π 多選「min(Q1, Q2)(s, a) − α ln π(a|s)」高的 a，其實就是把 π 推向 softmax(Q/α)。

- 離散 action：可以直接算每個 action 的機率。
- 連續 action（例如關節力矩）：把動作寫成 a = tanh(μ(s) + σ(s)·ε)，ε 是標準常態雜訊。這樣梯度可以從 Q 經過 a 傳回 μ、σ（reparameterization）。這和趙書 10.4 節 deterministic policy gradient「梯度穿過 Q」的想法相同，和第07—09課用 ∇ln π 的做法不一樣。
- 本課實驗用表格，直接把 π 設成 softmax(min Q / α)，等於 actor 每一步都學到最好。

**第 8 步：完整的 SAC 迴圈。**

```
 ┌─> 1. 用目前的 π 和環境互動一步，把 (s, a, r, s', 是否結束) 存進 replay buffer
 │      （一開始的 learning_starts 步用隨機動作）
 │   2. 從 buffer 隨機抽一批（新舊資料混在一起）
 │   3. 兩個 Q 都往 y 學；y 用 min(Q̄1, Q̄2) 和目前的 π 算（模組 A 第 6 步）
 │   4. actor：把 π 推向 softmax(min Q / α)
 │   5. 自動調 α（第 6 步）；soft update：Q̄ ← Q̄ + τ(Q − Q̄)
 └── 回到 1。資料不丟，buffer 滿了才擠掉最舊的
```

| | PPO（第09課） | SAC |
|---|---|---|
| 資料 | 一輪用 epochs 次後丟掉（on-policy） | 存進 buffer 反覆用（off-policy），每筆約 256 次 |
| 舊資料怎麼處理 | ratio＋clip | critic 學 Q(s, a)，不需要 ratio |
| critic | V(s)，returns 每輪固定 | 兩個 Q(s, a)，取 min；soft target τ = 0.005 |
| entropy | loss 裡的小獎勵（`entropy_coef` 0.01） | 寫進目標，α 自動調 |
| 適合 | 模擬器開幾千個環境，資料便宜 | 資料貴，每一筆都要多用 |

Stable-Baselines3 的 SAC 預設（2026-11-11 核對）：`buffer_size=1000000`、`learning_starts=100`、`batch_size=256`、`tau=0.005`、`gamma=0.99`、`train_freq=1`、`gradient_steps=1`、`ent_coef='auto'`、`target_entropy='auto'`、`learning_rate=3e-4`。Isaac Lab 官方文件列出的整合函式庫裡，只有 RL-Games 有 SAC；rsl_rl 提供的是 PPO、distillation 等。

## 核心實驗（約 30 分鐘）：只改 Q 的數量

設定：A 可以選 left（結束，0 分）或 right（走到 B，0 分）。B 有 8 個 action，選哪個都結束，reward 平均 −0.1，外加 ±1.5 的均勻雜訊。α = 0.01，policy 幾乎只選估計最高的。每走一步，每個 Q 從 buffer 抽 4 筆更新；τ = 0.05；400 回合；10 個 seed。真值：Q(A, left) = 0，Q(A, right) = −0.078，所以幾乎都該走 left。

- `single`：只有一個 Q，算 target 時直接用它。
- `twin`：兩個 Q，各從 buffer 抽自己的一批；算 target 時每個 action 取兩者較小的那個。

1. **讀程式**：`Agent.learn` 裡哪一行從 buffer 抽資料？哪一行算 target？twin 時 target 用的是什麼？soft target update 在哪幾行？
2. **先預測**：single 和 twin 哪個比較常走 right？V(B) 的估計會高於還是低於真值 −0.079？
3. **運行並記錄**：兩段回合（1—100、101—400）各自的走 right 比例、V(B) 估計、Q(A, right) 估計；每個 seed 在第 101—400 回合走 right 的比例。前 100 回合兩種都常走 right：B 的 action 還沒試過時估計是 0，比真值 −0.1 高。重點看後 300 回合。
4. **讀 B 段**：為什麼 Q1 各 action 的平均 ≈ −0.104（不偏），max 卻是 +0.451？取 min(Q1, Q2) 之後為什麼降到 +0.130？+0.130 還是比真值高，這說明什麼？
5. **只改一個條件**：`N_B = 32`。先預測 single、twin 各會怎樣（B 段也會跟著變），再運行。做完改回 8。

## 擴充實驗（約 25 分鐘）：只改 α

設定（陷阱與寶藏）：在 S 可以選 safe（結束，+1）或 go（走到 C1）。C1—C4 每一格都可以選 quit（結束，0 分）或 go（走到下一格）；在 C4 選 go 拿到寶藏 +3。一直走到寶藏才是最好的（實際 reward 3，比 safe 的 1 多）。twin Q，300 回合，10 個 seed。

1. **先預測**：α = 0.05、0.3、1、3 裡，哪個會卡在 safe？哪個最好？α 很大會怎樣？
2. **運行並記錄**：C 段表格（拿過寶藏的 seed 數、最後 100 回合平均 reward 與最小值、最後的 π(go|S)）。
3. **只改一個條件**：`REWARD_SCALE = 10`（所有 reward 乘 10）。先預測最好的 α 會變成哪一個，再運行。表裡有一列的數字剛好是原本某一列的 10 倍，是哪兩列？為什麼？做完改回 1。
4. **證據範圍**：能不能說「SAC 的 α 設 1 最好」？

## 綜合小測（5 題，集中提交）

1. **計算（twin target）**：γ = 0.99，α 很小（V 取 min 之後的最大值）。一筆資料 (s, a, r = 0.5, s')，s' 沒有結束，有三個 action：Q̄1(s', ·) = (1.0, 0.2, 0.6)，Q̄2(s', ·) = (0.7, 0.5, 0.9)。求（a）只用 Q̄1 的 target；（b）twin 的 target。
2. **計算（α 與 policy）**：兩個 action，Q = (2, 1)。查表值：e² = 7.389，e^0.5 = 1.649。（a）α = 0.5 與 α = 2 時，π(第一個) 各是多少？（b）α = 0.5 時 entropy = 0.365。如果目標 entropy H̄ = 0.5，自動調整會讓 α 變大還是變小？為什麼？
3. **解釋**：機器人上個月收集的資料，今天 SAC 的 critic 還能直接用嗎？PPO 的 actor 呢？各自為什麼？舊資料有沒有可能造成麻煩？
4. **證據範圍**：核心實驗中，10 個 seed 平均起來 twin 比較少走 right。但 seed 4 是 single 0.00、twin 0.43。另外導師試過 `N_B = 2`：第 101—400 回合 single 的 V(B) 估計約 −0.166，twin 約 −0.262，真值 −0.093。能不能說「twin 永遠比較準」？twin 的代價是什麼？
5. **遷移**：（a）讓真機器人學抓東西（每小時只能收幾千步），和在 Isaac Lab 開 4096 個環境學走路，你會各先考慮 PPO 還是 SAC？為什麼？（b）如果 SAC 訓練時 entropy 很早就掉得很低、policy 不再嘗試新動作，你會先檢查哪些設定？

## 提交格式

~~~text
實際用時（未計時可寫未計時）：
結果來源：瀏覽器 / 電腦本地 / 分析導師輸出
閉卷回憶 1—4：
模組 A 檢查：
核心實驗預測與記錄（兩段回合的走 right 比例、V(B)、Q(A,right)；每個 seed）：
B 段的解釋：
N_B = 32 的預測與結果：
擴充實驗預測與記錄（α、拿過寶藏、平均 reward、π(go|S)）：
REWARD_SCALE = 10 的預測、結果與解釋：
小測 1—5：
本課結論的適用範圍：
仍不確定的地方：
~~~

## 課堂補充（上課時的追問與重講）

### 補充 1：為什麼 critic 用舊資料不需要 ratio（一句話版）

critic 把 action 當條件，資料裡各個 action 出現多少次都無所謂；actor 的梯度是對 action 取平均，頻率決定答案。只說「SAC 是 off-policy、PPO 是 on-policy」是名稱，不是理由。

但 off-policy 不代表舊資料沒有問題：buffer 裡的 state 分佈來自以前的 policy（第06課的更新分佈），現在 policy 會去的地方資料可能很少；環境本身改變（磨損、重新校正、物體換位置）時，「做了 a 之後 r、s' 由環境決定」的前提就不成立。

### 補充 2：讀程式時分清兩行

`cache[s] = soft_value(...)` 算的是 V(s')；真正的 target 是 `y = r if s2 is None else r + GAMMA * v_target(s2)`。

### 補充 3：N_B = 32 為什麼更糟

| 第 101—400 回合 | single | twin | 真值 |
|---|---|---|---|
| 走 right 的比例 | 0.92 | 0.82 | 幾乎 0 |
| V(B) 估計 | +0.207 | +0.121 | −0.065 |
| B 段 max 的平均 | +0.678（8 個 action 時 +0.451） | +0.325（8 個時 +0.130） | −0.1 |

從 32 個帶雜訊的估計裡挑最大的，比從 8 個裡挑更容易挑到極端偏高的那個。twin 仍有幫助，但壓不住了。反過來，`N_B = 2` 時兩者都低估，twin 低估更多（V(B) −0.262 vs single −0.166，真值 −0.093）：低估會讓 agent 不再去 B，沒有新資料，估計就改不回來。

### 補充 4：α 太大時為什麼 π(go|S) 很高、reward 卻很低

α = 3 時 π(go|S) = 0.82，但 1.149 ≈ 0.18 × 1 + 0.82 × P × 3，算得進鏈之後走到寶藏的機率 P 只有約四成（α = 1 時約八成）。鏈上每一格都有選擇、都能拿 entropy 加分，所以 go 看起來很誘人，進去之後卻常常 quit：maximum entropy 偏好選項多的路。

### 補充 5：兩個 action 的 softmax 只看差值

$$
\pi(\text{第一個})=\frac{1}{1+e^{-(Q_1-Q_2)/\alpha}}
$$

Q = (2, 1)：α = 0.5 時差值 ÷ α = 2，π = e²/(e² + 1) = 0.881；α = 2 時差值 ÷ α = 0.5，π = e^0.5/(e^0.5 + 1) = 0.622。

### 補充 6：entropy 太早掉很低時檢查什麼

entropy 太低代表 α 相對太小（不是太大）。依序檢查 `ent_coef`（固定值太小？有沒有用 `'auto'`？）、`target_entropy`、α 的初始值與學習率、reward 的尺度（reward 大，同樣的 α 等於變小）、`learning_starts`。

## 資料與課程對應

- Haarnoja et al., 2018, *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*，[arXiv:1801.01290](https://arxiv.org/abs/1801.01290)。
- Haarnoja et al., 2018, *Soft Actor-Critic Algorithms and Applications*，[arXiv:1812.05905](https://arxiv.org/abs/1812.05905)（自動調 α）。
- Fujimoto et al., 2018, *Addressing Function Approximation Error in Actor-Critic Methods*，[arXiv:1802.09477](https://arxiv.org/abs/1802.09477)（clipped double Q）。
- Christodoulou, 2019, *Soft Actor-Critic for Discrete Action Settings*，[arXiv:1910.07207](https://arxiv.org/abs/1910.07207)（離散 action 版本，本課實驗的寫法）。
- Sutton & Barto 第 2 版 6.7 節（maximization bias 與 double learning）。
- 趙世鈺《強化學習的數學原理》8.4 節（experience replay、target network）、10.4 節（deterministic actor-critic，梯度穿過 Q）。SAC 本身不在教材內（10.5 節提到 SAC、TD3）。[作者官方倉庫](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)
- Stable-Baselines3 SAC：[文件](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)與 `sac.py` 原始碼（2026-11-11 核對）。
- Isaac Lab：[Reinforcement Learning](https://isaac-sim.github.io/IsaacLab/main/source/concepts/reinforcement_learning.html)（2026-11-11 核對，整合函式庫與演算法列表）。

兩個實驗的環境與數字為本課自編的教學設定。
