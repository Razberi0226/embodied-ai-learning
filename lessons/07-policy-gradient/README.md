# 第 07 課：policy gradient——直接學 policy

核心問題：之前的 policy 都是先學 q、再從 q 挑 action。讓 policy 自己帶參數、直接調參數讓平均 return 變大時，該往哪個方向調？只靠試出來的結果，算得出這個方向嗎？

學習日期：2026-11-05 至 2026-11-08。計畫約 120 分鐘，含休息。

## 學習順序

| 順序 | 材料 | 內容 |
|---|---|---|
| 1 | [講義](notes/policy-gradient-reinforce.md) | 模組 A：softmax policy、目標 J、精確梯度 ∂J/∂θ_a = π(a)(r̄(a) − J)。模組 B：∇π = π∇ln π、REINFORCE、softmax 的 ∇ln π、多步 episode 用 q_t、為什麼是 on-policy、和 PPO actor 的關係 |

## 實驗

| 腳本 | 說明 |
|---|---|
| [policy_gradient_bandit.py](experiments/policy_gradient_bandit.py) | 兩個 action 的 bandit。A 段驗證一次更新的三種可能、平均等於 α∇J；核心實驗只改「梯度怎麼來」（精確 vs REINFORCE，5 個 seed）；擴充實驗只改 `OFFSET`。[參考輸出](experiments/reference_output.txt) |

```bash
python lessons/07-policy-gradient/experiments/policy_gradient_bandit.py
```

亂數產生器寫在腳本裡，所以在任何 Python 環境跑出來的數字都一樣。

## 本課要點

- REINFORCE 用一次抽樣的 r·∇ln π(a) 代替梯度，平均起來等於 ∇J，不需要知道每個 action 的平均 reward。
- 抽樣必須照目前的 π（on-policy）。改成永遠一半一半抽、又不修正權重，會停在 π(R) ≈ 2/3，而不是學到接近 1。
- 每個 reward 加上同一個 `OFFSET`，期望更新不變，因為 Σπ(a)∇ln π(a) = 0；但每一步的推力變大。`OFFSET` = 10 時，seed 1—5 有 2 個卡在 L（seed 6—10 是 3 個）。
- 減掉接近平均的數（`OFFSET` = −1.5）會讓結果之間的差距變小，這就是 baseline 的想法。PPO 的 advantage = return − value，用 critic 當 baseline，是第 08 課的主題。
- rsl_rl PPO 的 actor loss 是 −advantage × ratio，θ = θ_old 時的梯度方向是 advantage × ∇ln π。

## 學習記錄

[第 07 課學習記錄](learning-notes.md)

## 教材對應

趙世鈺《強化學習的數學原理》第 9 章 9.1、9.3、9.4（REINFORCE，Algorithm 9.1）；Sutton & Barto 第 2 版 2.8 節（Gradient Bandit）與 13.3 節（REINFORCE）；Williams (1992)。兩個 action 的環境與數字為本課自編的教學設定。
