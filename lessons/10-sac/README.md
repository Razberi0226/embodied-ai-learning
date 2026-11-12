# 第 10 課：SAC——把舊資料存起來一直用，再把「保持隨機」寫進目標

接著[第 09 課](../09-ppo/README.md)。核心問題：PPO 的每批資料只在一輪裡用，之後就丟掉。資料很貴的時候，能不能把舊資料存起來一直用？為什麼 SAC 用舊資料不需要 ratio？Q 為什麼會越學越偏高，兩個 Q 怎麼把它壓下來？α 這個「隨機程度」的旋鈕到底在調什麼？

學習日期：2026-11-11。計畫約 120 分鐘，含休息。

## 學習順序

| 順序 | 材料 | 內容 |
|---|---|---|
| 1 | [講義](notes/sac-twin-q-entropy.md) | 模組 A：replay buffer、critic 用舊資料為什麼不需要 ratio、maximization bias、twin Q（clipped double Q）、soft target。模組 B：entropy 與 α、π ∝ exp(Q/α)、soft V、α 與 reward 尺度、自動調 α、actor 的 reparameterization、完整 SAC 迴圈、與 PPO 對照、Stable-Baselines3 預設。課堂補充：off-policy 的理由與舊資料的代價、N_B 的影響、α 太大時的行為、softmax 只看差值、entropy 太低時檢查什麼 |

## 實驗

| 腳本 | 說明 |
|---|---|
| [sac_twin_q_alpha.py](experiments/sac_twin_q_alpha.py) | A 段核心實驗：maximization bias 的兩步 MDP（B 有 8 個 action），只改 Q 的數量（single、twin），10 個 seed。B 段：不訓練，只抽樣，放大看 max 怎麼造成高估、min(Q1, Q2) 怎麼壓下來。C 段擴充實驗：「陷阱與寶藏」，只改 α（0.05、0.3、1、3）。[參考輸出](experiments/reference_output.txt) |

```bash
python lessons/10-sac/experiments/sac_twin_q_alpha.py
```

為了看清楚原理，實驗用表格代替神經網路，policy 直接取 softmax(min Q / α)。

## 本課要點

- critic 學 Q(s, a)，action 是條件，r 與 s' 由環境決定，s' 之後用目前的 π 算，所以用舊資料不需要 ratio；但 buffer 的狀態分佈和環境變化仍會造成麻煩。
- 取 max 會挑到雜訊偏高的估計：B 段 8 個 action 真值都是 −0.1，max 的平均是 +0.451；取 min(Q1, Q2) 後降到 +0.130。
- 核心實驗第 101—400 回合：single 走 right 0.43、V(B) 估計 +0.069；twin 0.20、−0.040（真值 −0.079）。`N_B = 32` 時兩者都嚴重高估；`N_B = 2` 時兩者都低估，twin 更多。
- SAC 的 policy 是 π ∝ exp(Q/α)：α 小接近貪心、α 大接近平均分配。陷阱與寶藏中 α = 0.05 卡在 safe（平均 reward 1.000），α = 1 最好（2.265），α = 3 太隨機（1.149）。
- policy 只看 Q/α：所有 reward 乘 10 時，α = 3 那一列和原本 α = 0.3 那一列完全相同。這就是 SAC 第二版改成依目標 entropy 自動調 α 的原因之一。

## 學習記錄

[第 10 課學習記錄](learning-notes.md)

## 教材對應

Haarnoja et al. (2018) SAC 與 SAC Algorithms and Applications；Fujimoto et al. (2018) clipped double Q；Christodoulou (2019) 離散 action 的 SAC；Sutton & Barto 第 2 版 6.7 節；趙世鈺《強化學習的數學原理》8.4 節、10.4 節（SAC 本身不在教材內）；Stable-Baselines3 SAC 文件與原始碼、Isaac Lab Reinforcement Learning 文件（2026-11-11 核對）。兩個實驗的環境為本課自編的教學設定。
