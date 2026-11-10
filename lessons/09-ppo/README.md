# 第 09 課：PPO——同一批資料用好幾次，但不能走太遠

接著[第 08 課](../08-actor-critic/README.md)。核心問題：REINFORCE 和 A2C 的每批資料只用一次就丟掉。收集資料要跑環境，很花時間，能不能同一批多用幾次？多用幾次會出什麼事，PPO 怎麼防？另外，GAE 的 λ 到底在調什麼？

學習日期：2026-11-10。計畫約 120 分鐘，含休息。

## 學習順序

| 順序 | 材料 | 內容 |
|---|---|---|
| 1 | [講義](notes/ppo-clip-gae.md) | 模組 A：重複使用資料、ratio = π_新 / π_舊、崩壞例子、clip 與 min、rsl_rl 的 surrogate 與 adaptive learning rate。模組 B：GAE 的遞迴、λ 的兩端（TD 與 MC）、完整的 PPO 迴圈、Isaac Lab 設定對照。課堂補充：batch／mini-batch／epoch 與 rsl_rl 的打亂切份、clip 之後推力為什麼是 0、GAE 權重與 critic 品質 |

## 實驗

| 腳本 | 說明 |
|---|---|
| [ppo_clip_gae.py](experiments/ppo_clip_gae.py) | A 段核心實驗：兩個 action 的 bandit，每輪 4 筆資料，只改更新方式（one_step、no_clip、clip），10 個 seed。B 段：seed 6 第 18 輪的崩壞放大，逐 epoch 的 ratio(L)。C 段擴充實驗：只改 GAE 的 λ，比較梯度估計的偏差、分散與 RMSE。[參考輸出](experiments/reference_output.txt) |

```bash
python lessons/09-ppo/experiments/ppo_clip_gae.py
```

## 本課要點

- 同一批資料用多次時，資料來自 π_舊，每筆要乘 ratio = π_新 / π_舊 修正。機率小的 action，ratio 能被放大到 1 / π_舊（例子裡 π_舊(L) = 0.0061，ratio(L) 長到 163）。
- no_clip 前期最快（第 5 輪平均 π(R) 0.956），也最危險：10 個 seed 裡崩壞 1 個，單輪最大下降 −0.99。clip 單輪最大下降 −0.17、沒有崩壞，第 30 輪平均 0.962。快和崩來自同一個原因：同一批資料推很多次。
- clip 的 min 只在「往有利方向走過頭」時把目標截平，那一筆的推力（斜率）變成 0，不是把推力限制在 1.2 倍。
- `CLIP_EPS` 調的是圍欄的位置：改成 0.5 時更快（第 30 輪 0.987），但單輪最大下降變成 −0.29。
- GAE：A_t = δ_t + γλ·A_{t+1}；λ = 0 是 TD，λ = 1 是 MC。最好的 λ 取決於 critic 品質：品質 1 → λ = 0，品質 0.5 → λ = 0.95，品質 0 → λ = 1。
- rsl_rl 一輪更新 `num_learning_epochs` × `num_mini_batches` 次；這段期間 returns 固定、value loss 也 clip，作用類似第 06 課的 target network。

## 學習記錄

[第 09 課學習記錄](learning-notes.md)

## 教材對應

Schulman et al. (2017) PPO；Schulman et al. (2016) GAE；趙世鈺《強化學習的數學原理》第 10 章 10.3.1（importance sampling）；rsl_rl `ppo.py`、`rollout_storage.py`（2026-10-28 版本核對）。bandit 與兩條路的環境為本課自編的教學設定。
