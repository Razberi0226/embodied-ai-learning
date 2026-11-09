# 第 08 課：actor-critic——讓 critic 學出 baseline

接著[第 07 課](../07-policy-gradient/README.md)。核心問題：第 07 課的 baseline 是手動選的常數。state 一多、每個 state 平常能拿到的 return 都不同時，baseline 該怎麼選？能不能不等 episode 結束就更新？

學習日期：2026-11-09。計畫約 120 分鐘，含休息。

## 學習順序

| 順序 | 材料 | 內容 |
|---|---|---|
| 1 | [講義](notes/actor-critic-a2c.md) | 模組 A：state 相依的 baseline、advantage A(s, a) = q(s, a) − V(s)、actor 與 critic 的分工。模組 B：TD 版 advantage δ = r + γV(s') − V(s)、A2C 的兩個更新、critic 落後時的偏差、TD 與 MC 的取捨、rsl_rl 的 `compute_returns` 對照 |

## 實驗

| 腳本 | 說明 |
|---|---|
| [actor_critic.py](experiments/actor_critic.py) | A 段：critic 落後時 TD 與 MC 的 advantage 對比。B 段核心實驗：兩個 state 的 bandit，只改 baseline（none、shared、per_state）。C 段擴充實驗：兩步 MDP 的 A2C，只改 critic 步長（0、0.1、0.5）。D 段：critic 落後的訓練軌跡。[參考輸出](experiments/reference_output.txt) |

```bash
python lessons/08-actor-critic/experiments/actor_critic.py
```

## 本課要點

- 每個 state 減自己的 V(s)，期望更新不變；共用一個 baseline 時，對 reward 高的 state 太小、對 reward 低的 state 太大，兩邊都吵。本例中 per_state 在兩組各 5 個 seed 下都沒有卡住。
- A2C 用 δ = r + γV(s') − V(s) 當 advantage；終止時沒有 V(s') 那一項，δ = r − V(s)。教材第 10 章的 δ 是「target − 估計」，和前幾課的 error 符號相反。
- critic 不學（步長 0）時，V 永遠是 0，shortcut 的 δ 為 0、safe 的 δ 為 1，actor 變得只看眼前 reward，5/5 停在 safe。
- critic 落後時，TD 的 advantage 會偏：D 段裡 B 早已學會選 good，V(B) 卻到第 150 個 episode 才轉正，這段時間 shortcut 的 δ 一直是負的。
- TD 把責任分給真正犯錯的那一步，但依賴 critic；MC 不偏，但會把後面所有步驟的運氣混進前面。PPO 的 GAE 用 λ 在兩者之間取捨（第 09 課）。

## 學習記錄

[第 08 課學習記錄](learning-notes.md)

## 教材對應

趙世鈺《強化學習的數學原理》第 10 章 10.1、10.2（Algorithm 10.2）；Sutton & Barto 第 2 版 13.4、13.5 節；Mnih et al. (2016)；Schulman et al. (2016) GAE。兩個 state 的 bandit 與兩步 MDP 為本課自編的教學設定。
