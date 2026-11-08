# 第 06 課：critic 怎樣學得穩——更新分佈與 target network

接著[第 05 課](../05-function-approximation/README.md)的 moving target 預習。核心問題：semi-gradient TD 的 target 會跟著參數一起動，什麼時候仍然收斂、什麼時候會越跑越遠？target network 能幫上什麼，又幫不上什麼？

學習日期：2026-11-04。計畫約 120 分鐘，含休息。本課起講義改用繁體中文，專業術語仍用英文。

## 學習順序

| 順序 | 材料 | 內容 |
|---|---|---|
| 1 | [講義](notes/update-distribution-and-target-network.md) | 模組 A：平均更新 w ← w + α(b − Aw)，用 trace 與 det 判斷穩定；on-policy 為何安全；deadly triad。模組 B：target network 的做法、作用與限制；DQN 的 hard update、SAC 的 soft update；rsl_rl PPO 的對照 |
| 1a | 講義末尾的「模組 A 重講版」 | 以第 05 課的「乘數」為主線、矩陣最後才出現的 8 步講法（第一版太密，依學習者回饋改寫） |
| 1b | 講義末尾的鏈狀圖 | T1、T2 是同一條路上的先後兩步，不是岔路 |

## 實驗

| 腳本 | 說明 |
|---|---|
| [update_distribution_target_network.py](experiments/update_distribution_target_network.py) | 核心實驗只改更新分佈 `P_T1`（50、80、95）；擴充實驗只改 `SYNC_EVERY`（1、20、100）。[參考輸出](experiments/reference_output.txt) |

```bash
python lessons/06-target-network/experiments/update_distribution_target_network.py
```

## 本課要點

- 更新時各個 transition 出現的比例（update distribution）決定 TD-Linear 穩不穩。本例 `P_T1` = 50 是 on-policy 比例；小 α 的平均分析預測 `P_T1` > 0.877 會發散。
- 只更新 T1 時，沿 (1, 1) 方向的誤差每步乘 1.07；終止的 T2 是固定 target 回歸，會縮小誤差。兩者的比例決定誰贏。
- 分析有兩個前提：把更新平均起來、α 很小。靠近臨界值時（`P_T1` = 90、α = 0.1）實際運行與分析不一致。
- target network 讓兩次同步之間回到固定 target 的回歸；`SYNC_EVERY` = 1 等於沒有 target network，太大則收斂慢。資料裡沒有 T2 時，target network 也補不出資訊。
- td_loss 衡量的是離 target 多遠，不是離真實 value 多遠：沒有 T2、`SYNC_EVERY` = 100 時，第 100 次的 td_loss 只有 0.0089，true_error 卻是 2.2。

## 學習記錄

[第 06 課學習記錄](learning-notes.md)

## 教材對應

趙世鈺《強化學習的數學原理》第 8 章 8.2.2、8.2.5、8.4.1；Sutton & Barto 第 2 版 9.4、11.2、11.3 節。兩個 state 的環境與數字為本課自編的教學設定。
