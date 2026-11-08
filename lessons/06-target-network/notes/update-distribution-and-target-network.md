# 第06課：critic 怎樣學得穩——更新分佈與 target network

發布日期：2026-11-04。接續第05課與 moving target 預習。第01—05課已完成，本課發布不代表完成。

核心問題：semi-gradient TD 的 target 會跟著 w 一起動，什麼時候這仍然能收斂，什麼時候會越跑越遠？target network 能幫上什麼忙，又幫不上什麼？

## 學習安排與交付

| 環節 | 預計分鐘 | 要完成的事 |
|---|---:|---|
| 閉卷回憶 | 5 | 4 題，含第05課留下的證據範圍題 |
| 模組 A 與 B | 30 | 更新分佈與穩定性；target network 的作用與限制 |
| 休息 | 10 | 離開螢幕 |
| 核心與擴充實驗 | 55 | 先預測，讀關鍵程式，再只改一個條件比較 |
| 綜合小測 | 15 | 5 題集中提交 |
| 學習記錄 | 5 | 結論的適用範圍與剩下的疑問 |

120 分鐘是計畫，不代表實際用時。卡住時先交卡住的位置。

- 輸入：第05課的線性 v̂、gradient step，以及預習的兩個 transition。
- 成果：一張更新分佈對照表、一張 target network 對照表、5 題答案、一段結論範圍說明。
- 運行：執行 [`update_distribution_target_network.py`](../experiments/update_distribution_target_network.py)，可對照[參考輸出](../experiments/reference_output.txt)。只用標準庫，不需要 GPU。
- 程式已由導師實際執行檢查；導師執行不算作你完成。

## 閉卷回憶（5 分鐘）

1. 本課程的 error 是誰減誰？gradient 和「下降方向」差一個什麼？
2. table TD 的 target 不動時，gap 每次更新乘以多少？
3. train loss 和共同 eval error 各衡量什麼？
4. 用自己的話寫一條第05課實驗「證明不了」的結論。

## 模組 A：更新分佈決定 TD-Linear 穩不穩

### A1. 回顧 TD-Linear

教材 8.2.2 的 TD-Linear（式 8.14）寫成「w 加上 α·(target − prediction)·φ」。本課程沿用 error = prediction − target、再減去 α·error·φ，兩者數值相同：

$$
e=\phi(s)^\top w-\big(r+\gamma\,\phi(s')^\top w\big),\qquad w\leftarrow w-\alpha\,e\,\phi(s).
$$

真正終止時沒有 bootstrap，把 φ(s') 當成 0。本課的 φ(x) = (x, 1)，即 v̂(x) = w₁x + w₀。

### A2. 把很多次更新平均起來看

把 e 展開：e = (φ − γφ')ᵀw − r。所以一次更新是

$$
w\leftarrow w-\alpha\,\phi(\phi-\gamma\phi')^\top w+\alpha\,r\,\phi .
$$

對「這次更新到哪個 transition」取平均，得到

$$
w\leftarrow w+\alpha\,(b-Aw),\qquad A=\mathbb E\big[\phi(\phi-\gamma\phi')^\top\big],\quad b=\mathbb E[r\,\phi].
$$

這就是教材 8.2.5 的式 (8.22)；教材用矩陣寫成 A = ΦᵀD(I − γP_π)Φ，D 的對角線放每個 state 出現的比例，意思相同。

令 w* 滿足 Aw* = b，誤差 δ = w − w* 每次變成 (I − αA)δ。這是第05課「每步乘一個乘數」的推廣：

- 第05課是固定 target，A 變成 E[φφᵀ]。訓練位置的 x 平均為 0 時它是對角矩陣，所以兩個方向各乘 1 − αc 和 1 − α。
- 現在 A 一般不是對角、也不對稱，要看它的 eigenvalues。

### A3. 2×2 的簡便判斷法

2×2 矩陣的兩個 eigenvalues 之和是 trace（對角線相加），乘積是 det。所以在 α 夠小時：

| trace | det | 結果 |
|---|---|---|
| > 0 | > 0 | 兩個 eigenvalue 的實部都 > 0：收斂 |
| < 0 | > 0 | 兩個實部都 < 0：發散 |
| 任意 | < 0 | 一正一負：有一個方向發散 |

另外，trace² − 4·det < 0 時 eigenvalues 是複數，w 會一邊收斂（或發散）一邊來回擺。

### A4. 例題：預習的兩組

兩種 transition：T1 是 x=1 → x=2、r=0；T2 是 x=2 → 終止、r=1。真實 value：v(2)=1，v(1)=0.9。

**只更新 T1（預習 A 組）。** φ = (1,1)，φ − γφ' = (1,1) − 0.9·(2,1) = (−0.8, 0.1)：

$$
A=\begin{pmatrix}1\\1\end{pmatrix}\begin{pmatrix}-0.8&0.1\end{pmatrix}
=\begin{pmatrix}-0.8&0.1\\-0.8&0.1\end{pmatrix},\quad \text{trace}=-0.7,\ \det=0 .
$$

非零的 eigenvalue 是 −0.7，方向是 (1,1)，也就是每次更新推動 w 的方向。這個方向每步乘 1 − α·(−0.7) = 1.07，正是預習時算出的 1.07。

**T1、T2 各一半（on-policy：每個 episode 兩個 state 各經過一次）。** T2 終止，φ' = 0，貢獻 φφᵀ：

$$
A=\tfrac12\begin{pmatrix}-0.8&0.1\\-0.8&0.1\end{pmatrix}+\tfrac12\begin{pmatrix}4&2\\2&1\end{pmatrix}
=\begin{pmatrix}1.6&1.05\\0.6&0.55\end{pmatrix},\quad \text{trace}=2.15,\ \det=0.25 .
$$

兩者都 > 0，所以收斂。

**一般的比例。** 設 T1 佔 p，T2 佔 1 − p：

$$
A(p)=\begin{pmatrix}4-4.8p&2-1.9p\\2-2.8p&1-0.9p\end{pmatrix},\qquad \text{trace}=5-5.7p,\qquad \det=p(1-p).
$$

### A5. 為什麼 on-policy 就穩

教材 8.2.5 證明：state 按 policy 實際出現的比例（stationary distribution d_π）被更新時，A 是 positive definite，TD-Linear 收斂到 w* = A⁻¹b，也就是 projected Bellman error 最小的解。本例是 episodic 任務，對應的比例就是 T1:T2 = 1:1。

換成別的比例時沒有這個保證。這通常發生在 off-policy：資料由 behavior policy 產生，或從 replay buffer 抽樣，比例不等於要評估的 policy。

**deadly triad**：function approximation、bootstrapping、off-policy 這三者同時出現，就可能發散（Sutton & Barto 11.3 節）。拿掉任何一個，本課的這種發散就消失：

| 拿掉 | 例子 |
|---|---|
| function approximation | table：每個 state 有自己的參數，改 v(1) 不會動到 v(2) |
| bootstrapping | 用 MC return 當 target：target 不含 w，回到第05課的固定 target 回歸 |
| off-policy 的更新分佈 | 按 on-policy 比例更新：A positive definite |

另一點：本例的直線能精確表示真實 value，所以只要 0 < p < 1，收斂點都是同一個 w* = (0.1, 0.8)，p 只影響穩不穩、快不快。直線表示不了真實 value 時，p 還會改變收斂到哪裡。

## 模組 B：target network——把 moving target 暫時凍住

### B1. 做法

教材 8.4.1 在 DQN 裡用兩組參數：main network 的 w，以及 target network 的 w_T。

1. TD target 用 w_T 算：r + γ·v̂(s', w_T)。
2. 照常用 gradient step 更新 w。
3. 每 C 次更新，令 w_T ← w。

兩次同步之間，target 完全不動，就是第05課的固定 target 回歸，它的 A 是 E[φφᵀ]，只要 α 夠小就穩。target 只在同步時跳一次。程式裡 `SYNC_EVERY = 1` 等於沒有 target network。

### B2. 幫得上和幫不上的地方

幫得上：把「一邊追、目標一邊跑」改成「先追一個不動的目標，追得差不多再換目標」。

幫不上的地方：

1. **資料裡沒有的資訊**：如果從來沒有 T2，v̂(2) 沒有任何錨點。每個階段都把 v̂(1) 擬合到 0.9·v̂(2)，卻同時把 v̂(2) 推得更高，target network 一樣會發散，只是慢一些。擴充實驗會跑這一組。
2. **一般情況沒有保證**：Sutton & Barto 11.2 節（含 Tsitsiklis 和 Van Roy 的反例）說明，即使每個階段都完整擬合，function approximation 加上 bootstrapping 仍可能發散。本例能成功，部分是因為直線能精確表示真實 value。
3. **C 太大**：每個階段的 target 都太舊，收斂變慢。

另外要注意：td_loss 量的是「離 target 多遠」，不是「離真實 value 多遠」，這是第05課 train loss 與 eval error 的延伸。

### B3. 實務上的兩種同步方式

- **hard update**：每 C 步直接複製。DQN（Mnih et al., 2015）用 C = 10000。
- **soft update**：每步只把一小部分移過去，w_T ← (1 − τ)w_T + τw。SAC（Haarnoja et al., 2018）用 τ = 0.005。

### B4. 和 Isaac Lab 常用的 PPO（rsl_rl）的關係

【原始碼】rsl_rl 的 PPO 在每個 iteration 開頭呼叫一次 `compute_returns()`，用收集 rollout 時存下的舊 values（加上 `detach()` 過的最後一步 value）算 GAE 和 returns。接著 `update()` 用這組固定的 `batch.returns` 跑 `num_learning_epochs`（預設 5）輪 mini-batch 更新。value loss 還可以相對舊 values 做 clipping。

【推斷】所以 PPO 的 critic 在一個 iteration 內追的是一個不動的 target，相當於每個 iteration 同步一次的 target network；而資料是當前 policy 剛收集的 rollout，接近模組 A 的 on-policy 比例。本課的兩種不穩定來源，PPO 的設計大致都避開了。訓練人形機器人時 value loss 偏高或不穩，更值得先查 target 的尺度（reward 大小、γ 與 episode 長度造成很大的 return）、observation 的尺度（第05課的 feature 尺度），以及 learning rate。

## 核心實驗（約 30 分鐘）：只改更新分佈

設定：α = 0.1，起點 w = (1, 0)，600 次更新，沒有 target network。`P_T1` 是每 100 次更新中 T1 的次數，用固定週期平均分散，不用隨機數，所以不需要 seed。真實 value 已知，`true_error` 是兩個 state 中較大的那個 |v̂ − v|。

1. **先預測**：用 A4 的公式算 P_T1 = 80、95 的 trace 和 det，判斷哪個會發散；80 會不會來回擺？
2. **讀程式**：`expected_matrix` 裡哪一行處理「終止時 φ' = 0」？`train` 裡哪一行決定 target 用 `w_target`？
3. **運行並記錄**：每組在第 50、200、600 次的 true_error 和 status。
4. **只改一個條件**：把 `CORE_P_T1` 改成 `(90,)`。先寫 A4 的分析會怎麼預測，再運行。結果和分析一致嗎？如果不一致，分析用了哪兩個實際程式沒有滿足的假設？

## 擴充實驗（約 25 分鐘）：只改 SYNC_EVERY

設定：P_T1 = 95 固定，1000 次更新，`SYNC_EVERY` = 1、20、100。

1. **先預測**：哪幾組會發散？
2. **運行並記錄**：每組最後的 true_error、td_loss 和 status。
3. **只改一個條件**：`EXT_P_T1 = 100`、`EXT_SYNC = (100,)`。先預測，再運行。比較第 100 次的 td_loss 和 true_error，解釋為什麼差這麼多。
4. **選做**：`EXT_SYNC = (50,)`。true_error 會不會隨更新次數單調下降？能不能從 1、20、50、100 的結果說「SYNC 越大越好」或「越小越好」？提示：T2 固定每 20 次出現一次，同步週期和這個週期的配合也會影響結果。

## 綜合小測（5 題，集中提交）

1. **計算**：用 trace = 5 − 5.7p、det = p(1 − p)，求 α 很小時 TD-Linear 收斂的 p 範圍。p = 0.6 收斂嗎？
2. **解釋**：為什麼加入 T2 這種終止的更新，就能讓 A 變穩？請用 A 由哪兩部分組成來說明。
3. **程式**：如果刪掉 `if k % sync_every == 0: w_target = w` 這兩行，target network 會永遠停在起點 (1, 0)。P_T1 = 95 時，w 最後會收斂嗎？收斂到的 v̂(1)、v̂(2) 是多少？和真實 value 一樣嗎？
4. **證據範圍**：核心實驗裡 P_T1 = 80 比 50 收斂得快。能不能說「off-policy 的更新分佈比 on-policy 好」？為什麼？至少說出一個本實驗沒有涵蓋到的條件。
5. **遷移到 PPO**：rsl_rl 每個 iteration 算一次 returns，再做 `num_learning_epochs × num_mini_batches` 次 gradient step。對照本課，這相當於 SYNC_EVERY 設成多少？它的資料比例最接近核心實驗的哪一組？value loss 降得很低，能不能說明 critic 已經準了？

## 提交格式

可以分兩批：先交閉卷回憶與核心實驗的預測，實驗結果和小測稍後一起交。

~~~text
實際用時（未計時可寫未計時）：
結果來源：本地運行 / 分析參考輸出
閉卷回憶 1—4：
核心實驗預測與記錄表（P_T1、第50/200/600次 true_error、status）：
P_T1=90 的預測、結果與解釋：
擴充實驗預測與記錄表（SYNC_EVERY、最後 true_error、td_loss、status）：
P_T1=100、SYNC=100 的結果與解釋：
小測 1—5：
本課結論的適用範圍：
仍不確定的地方：
~~~

完成以實際計算、程式解釋、對照與證據範圍為準；只說看懂或由導師代為運行，不算完整驗收。

## 資料與課程對應

- 趙世鈺《強化學習的數學原理》第 8 章：8.2.2（TD-Linear）、8.2.5（A、b、stationary distribution、projected Bellman error）、8.4.1（main network、target network、experience replay）。[作者官方倉庫](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)
- Sutton & Barto《Reinforcement Learning: An Introduction》第 2 版：9.4 節（線性 TD 的 fixed point 與 on-policy 穩定性）、11.2 節（off-policy 發散的例子，其中 w → 2w 的例子和本課只更新 T1 的情形幾乎一樣）、11.3 節（deadly triad）。[線上版](http://incompleteideas.net/book/the-book-2nd.html)
- Mnih et al., 2015, *Human-level control through deep reinforcement learning*, Nature.
- Haarnoja et al., 2018, *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*, [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)。
- rsl_rl：[`rsl_rl/algorithms/ppo.py`](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py) 的 `compute_returns` 與 `update`（2026-10-28 版本核對）。

兩個 state 的環境、數字與實驗為本課自編的教學設定，不聲稱原樣來自趙老師課程。

## 附：模組 A 重講版（2026-11-04，依學習者回饋改寫）

原本的模組 A 直接從矩陣講起，太跳。這版只用第05課的「乘數」，一步一步推，矩陣最後才出現，只是用來記帳。

**要回答的問題**：預習時，同一個更新公式，只更新 T1 會越跑越遠，T1、T2 輪流更新卻會收斂。為什麼？能不能不跑程式就先判斷？

1. **工具還是乘數。** 第05課：誤差每步乘 (1 − α) 或 (1 − αc)。絕對值 < 1 縮小、收斂；> 1 放大、發散。
2. **量離正確答案多遠。** 正確答案 w* = (0.1, 0.8)，此時 v̂(1)=0.9、v̂(2)=1。令 d = w − w*。在 w* 處兩種 error 都是 0，所以 error 只由 d 決定：T1 的 e = −0.8·d₁ + 0.1·d₀；T2 的 e = 2·d₁ + d₀。每次更新 d ← d − α·e·φ，T1 的 φ = (1,1)，T2 的 φ = (2,1)。
3. **只做 T1 為什麼放大。** T1 永遠沿 (1,1) 推。設 d = t·(1,1)，e = −0.7t，更新後 d 變成 1.07 倍。原因：沿 (1,1) 改 w，v̂(1) 漲 2 份，target 0.9·v̂(2) 漲 2.7 份。
4. **只做 T2 為什麼縮小。** T2 的 target 固定為 1，就是第05課的固定 target 回歸。設 d = t·(2,1)，e = 5t，更新後 d 變成 0.5 倍。但 T2 只管 v̂(2)，管不到 v̂(1)。
5. **混在一起，比例決定誰贏。** T1 把 v̂(1) 綁到 0.9·v̂(2)，有放大的傾向；T2 把 v̂(2) 釘在 1，有縮小的作用。兩者的比例就是更新分佈（P_T1）。
6. **把平均一次更新記成一張表 A。** 表的第 i 列 = 推的方向 φ 的第 i 個分量 × e 的係數。T1：[[−0.8, 0.1], [−0.8, 0.1]]；T2：[[4, 2], [2, 1]]。T1 佔 p 時，A(p) = p·(T1 的表) + (1 − p)·(T2 的表)。
7. **從 A 看乘數。** 一個數字時乘數是 1 − α·a，要 a > 0。2×2 的表有兩個特殊方向（eigenvector），各有自己的 a（eigenvalue），兩個都要 > 0。兩個 a 的和 = trace，乘積 = det，所以 trace > 0 且 det > 0 就收斂。驗算：p = 1 時 trace = −0.7、det = 0，兩個 a 是 0 和 −0.7，−0.7 的方向就是第 3 步的 (1,1)，乘數 1.07；p = 0.5 時 trace 2.15、det 0.25，收斂。
8. **為什麼 on-policy 安全。** 照 policy 跑 episode，T1、T2 各一次，p 自然是 0.5。教材 8.2.5 證明按 policy 實際比例更新時，線性 TD 一定收斂；刻意多更新 T1（off-policy、replay 抽樣不均）就可能發散。

一句話：放大來自「target 裡有 v̂（bootstrapping）＋共用參數（function approximation）」，縮小來自 target 固定的終止更新，比例決定誰贏。

## 附：T1、T2 是同一條路上的兩步，不是岔路（2026-11-04 補充）

```
state 1 ──T1（reward 0）──> state 2 ──T2（reward 1）──> 終止
```

每個 episode 都先走 T1、再走 T2，沒有選擇。P_T1 不是選路的機率，而是「每 100 次更新裡拿 T1 的紀錄來更新幾次」。照 policy 跑出來的資料必然是 T1、T2 各一半（P_T1 = 50，on-policy）；其他比例只能來自重抽、丟棄或別的 policy 的資料（off-policy）。
