# 第09課學習記錄：PPO

日期：2026-11-10（當天發布、當天完成）。
結果來源：學習者在瀏覽器內的 Python 環境運行 A—C 段及 CLIP_EPS=0.5、CRITIC_QUALITY=1.0／0.0、BATCH=16 變體，預設輸出與導師參考輸出一致。
實際用時：未計時。

## 結論：第09課完成

| 部分 | 結果 |
|---|---|
| 閉卷回憶 | δ 方向（target − 估計）與終止 δ = r − V 正確，非終止 target 未寫完整 r + γV(s')；第07課第3題只說「∇J 不一樣」，已補後果＝停在 π(R) ≈ 2/3、乘 ratio 後變回真 ∇J；critic 落後＝偏差：正確；target network 想法正確，PPO 對應誤答成 GAE，已補「returns 每輪固定（sync＝epochs×mini-batches）＋ value clip ±0.2」 |
| 追問：batch／mini-batch／epoch | 已講：mini-batch 份數 1＝整批、份數＝筆數＝逐筆；本課 BATCH=4 是筆數、不切；推力取平均；各環境先各自沿時間算 GAE 再攤平、整批 randperm 後切份（rollout_storage.py 第 232 行）；份數×epoch＝一輪更新次數＝policy 能離 π_舊 多遠 |
| 模組 A 檢查 | 第5題 ratio = 1.5 正確，但以為 clip 後「以 1.2 倍推」；經系統重講（目標值 vs 推力、連鎖律隱藏因子：clip 對 ratio 的斜率 1→0、min 只擋有利方向）後，自我檢查 a 對、b c 錯（忘乘 A、以為出界就截），重練 d e 全對；第6題（π_舊 小 → ratio 爆）正確 |
| 模組 B 檢查 | B1：λ = 0／0.5／1 的 A_0 = −1／0.8／2.6 全對，看出 critic 不準時 λ 會改變推或壓；B2：λ=1 returns = G_t、λ=0 returns = TD target：全對 |
| 核心實驗 | 讀程式正確；預測「clip 最快」錯，實際 no_clip 前期最快（第5輪 0.956）且 1/10 崩（單輪 −0.99），快與崩同一機制；B 段 ratio(L) 上限 1/0.0061 ≈ 164：正確；CLIP_EPS=0.5 預測變快：正確，漏了代價（單輪最大下降 −0.17 → −0.29、圍欄退到 1.5）；「CLIP_EPS 在調圍欄範圍」：正確 |
| 擴充實驗 | 品質 0.5：λ = 0.95 RMSE 最小、偏差隨 λ 減小、分散隨 λ 增大的解釋正確；品質 1.0：預測最佳 λ = 0.5 錯（沒偏差時不必付雜訊代價），看結果後解釋正確；品質 0.0：預測 λ=0 偏差 ≈ −0.644、最佳 λ = 1：全對；λ=1 那列在三種品質下完全相同（V 抵消），−0.019 為抽樣誤差（≈ 0.340/√300） |
| 小測 1 | 四種 clip 情況（甲、乙、min、推不推）全對 |
| 小測 2 | A_2 = 2、A_1 = 0.44 正確；A_0 寫成 0.5368，漏乘 δ_1 的 γλ，應為 0.5 + 0.72 × 0.44 = 0.8168 |
| 小測 3 | 方向對；已補：π(L) ≈ 0.998 時一批 4 筆全是 L 的機率 ≈ 99.4%，reward 全 1 → advantage 全 0 → 推力 0；剩下 12 輪裡至少抽到一次 R 的機率只有約 7%，抽到也有一半拿 0 分被再壓；entropy 在 surrogate 推力為 0 時仍有推力，把 policy 拉回分散（「保證」過強，係數 0.01 只是拉力） |
| 小測 4 | 「1/10 不能說 10%」正確；BATCH=16 後 no_clip 不崩，解釋「學得多、更可靠」方向對，已補：16 筆裡 R 全拿 0 分的機率 ≈ 0.5^15，誤導的一批幾乎不會出現，崩壞是小批次運氣觸發的 |
| 小測 5 | (a) 寫成 num_steps_per_env 與 num_mini_batches，應為 num_learning_epochs × num_mini_batches（num_steps_per_env 決定收多少，不是用幾次）；(b) clip_param、desired_kl：正確，可再加 learning_rate、更新次數；(c) 只寫 clip_param 太大或 batch 太小，已補曲線：Loss/learning_rate、Loss/entropy、Loss/value、Loss/surrogate、Policy/mean_std |

## 待複習（非阻塞）

1. clip 後的推力：min 選中截過的那一項且已往有利方向走過頭 → 斜率 0，不是「以 1.2 倍推」。
2. 目標值要乘 A；負數取 min 要小心（−2.6 < −2.4）。
3. GAE 展開時每一項都要乘 (γλ)^k；用倒推 A_t = δ_t + γλ·A_{t+1} 較不易錯。
4. 「同一批資料用幾次」＝ num_learning_epochs × num_mini_batches；num_steps_per_env × num_envs 是一批有多大。
5. 預測時把代價也寫出來：更快的設定通常單輪風險更大。

## 下一課入口（L10）

SAC：off-policy、replay buffer、entropy（maximum entropy RL）、twin Q、soft target（τ = 0.005），對照 PPO 的 on-policy 與 clip。
