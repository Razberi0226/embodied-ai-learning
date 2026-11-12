# 第10課學習記錄：SAC

日期：2026-11-11（當天發布、當天完成）。
結果來源：學習者在瀏覽器內的 Python 環境運行 A—C 段及 REWARD_SCALE=10 變體，預設輸出與導師參考輸出一致；N_B=32 的表格數字由導師運行提供（學習者有運行並描述結果）。
實際用時：未計時。

## 結論：第10課完成

| 部分 | 結果 |
|---|---|
| 閉卷回憶 | clip 三步法（甲 2.8、乙 2.4、不推）：全對，第09課誤解已修正；同一批資料用幾次＝`num_learning_epochs` × `num_mini_batches`：全對；Q-learning 能 off-policy：大意對，已補「a 是給定條件、下一步用目標 policy 的 max」；target network 與 hard／soft update：對 |
| 模組 A 檢查 | twin target：min = (0.1, −0.2)、V(B) = 0.1、y = 0.09，只用 Q̄1 時 y = 0.36：全對；為什麼不用 ratio 只答「off-policy／on-policy」（只有名稱），已補：critic 把 action 當條件，actor 的梯度對 action 取平均 |
| 模組 B 檢查 | soft V = 0.731 + 0.582 = 1.313 > max Q（entropy 加分）：全對；V(B) = −0.079 高於 −0.1 是 entropy 加分、α = 0 時為 −0.1：全對 |
| 核心實驗 | 讀程式大致對（target 行應為 `y = r if s2 is None else r + GAMMA * v_target(s2)`，soft_value 那行是算 V）；預測 single 較常走 right、V(B) 高估：正確；第101—400回合 single 走 right 0.43、V(B) +0.069，twin 0.20、−0.040（真值 −0.079）；B 段解釋前三點正確，「高估每個 action」已修正為「每個 action 不偏、取 max 後的 V 偏高，twin 減少但沒消除」；N_B = 32 結果描述對（single 0.92、twin 0.82），未寫預測，已補解釋 |
| 擴充實驗 | 預測 0.05 卡住正確、最佳 α 猜 3（實際 1）；α = 3 時 π(go\|S) 0.82 但進鏈後到寶藏約四成，已補「maximum entropy 偏好選項多的路」；REWARD_SCALE = 10 預測 α = 3 最好：正確，並指出新 α = 3 列與舊 α = 0.3 列完全相同（policy 只看 Q/α）；證據範圍：正確 |
| 小測 1 | 1.49、1.193（列式正確） |
| 小測 2 | (a) 分子對、分母錯：應為 e⁴/(e⁴ + e²) = 0.881、e¹/(e¹ + e^0.5) = 0.622（只看 Q 差 ÷ α）；(b) entropy 0.365 < 0.5 → α 變大：正確 |
| 小測 3 | SAC critic 能用、PPO actor 不能，理由「條件 vs 分佈平均」：正確；認為 SAC 舊資料不會有麻煩：不完整，已補 buffer 的狀態分佈與現在 policy 不同（第06課更新分佈）、環境本身改變時舊資料失效 |
| 小測 4 | 不能、twin 的代價是低估：正確；已補 N_B = 2 時 twin 低估更多，低估會讓 agent 不再去那裡、資料不更新（悲觀陷阱），以及 seed 間差異大 |
| 小測 5 | (a) 真機器人選 SAC、理由正確，Isaac Lab 4096 環境未答（應為 PPO：資料便宜、平行、穩定）；(b) 答「α 太大」方向相反：entropy 太低代表 α 相對太小，已補檢查 ent_coef／target_entropy／α 初值與學習率／reward 尺度／learning_starts |

## 待複習（非阻塞）

1. softmax 只看差值：π(第一個) = 1 / (1 + e^{−(Q1 − Q2)/α})。
2. entropy 太低 → α 相對太小（或 target_entropy 太低、reward 太大），不是 α 太大。
3. off-policy 不等於舊資料沒問題：資料分佈和環境變化仍會造成麻煩。
4. 讀程式時分清「算 V(s') 的行」和「算 target y 的行」。
5. 做「只改一個條件」時先寫預測。

## 下一步入口

PPO、SAC 主線（第06—10課）完成。後續可依用戶需要選擇：分析其 Isaac Lab PPO 訓練曲線與設定；補半課 PyTorch 基礎後做神經網路版 PPO／SAC；或進入具身智能主題（domain randomization、sim2real、imitation learning）。
