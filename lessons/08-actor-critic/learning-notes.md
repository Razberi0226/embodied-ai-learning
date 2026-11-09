# 第08課學習記錄：actor-critic

日期：2026-11-09（當天發布、當天完成）。
結果來源：學習者在瀏覽器內的 Python 環境運行 A—D 段，輸出與導師參考輸出一致；seed 6—10 的對照由導師運行提供。
實際用時：未計時。

## 結論：第08課完成

| 部分 | 結果 |
|---|---|
| 閉卷回憶 | 終止 target = r、與 target network 無關：第三次終於答對；REINFORCE 為何 on-policy 只答到 ∇ln π 形式，已補「抽樣頻率提供 π(a) 權重」；OFFSET 期望不變、推力大：正確；n 越大越接近 MC：精神正確 |
| 模組 A 檢查 | V(X)=11.5 時 L 拿 11 分 advantage −0.5（壓下）、共用 b=6.5 時 +4.5（推上）；V(Y) 由 1.5 更新為 1.75、target 是 r：全對 |
| 模組 B 檢查 | S 的 δ=1.3、MC advantage=−9.5 正確；B 的 δ 漏減 V(B)（應為 −12）；判斷「MC 較公平」相反，已講解 TD 把責任分給真正犯錯的一步、critic 落後時才偏 |
| 核心實驗 | 預測 none 的 X 最易卡、shared 連累 Y：正確；shared 的 b 約 6—8、critic V(X)≈12、V(Y)≈2.5：正確；seed 6—10 結論相同（none X 2/5、shared X、Y 各 1/5、per_state 0/5） |
| 擴充實驗 | ALPHA_CRITIC=0 預測會學到 shortcut 錯誤：V≡0 時 shortcut δ=0、safe δ=1，S 學成 safe（短視），5/5 停 safe；D 段 V(B) 第150個 episode 轉正正確，但 shortcut 的 δ 全程為負（−1.9 到 −0.34），學習者答正；0.1 vs 0.5 不能下結論：正確 |
| 小測 1 | 把 V(B) 代成 0，δ_S 應為 0.9×1.5−1.2=+0.15、V(S)→1.215；δ_B=0.5 正確；B 選 bad 時 δ_S 不變、δ_B=−11.5：正確 |
| 小測 2 | 方向對；已補每個 state 內 Σπ∇ln π·V(s)=0，以及共用 b 對 X 太小、對 Y 太大兩邊都吵 |
| 小測 3 | 對應終止時沒有 V(s')：正確；改成一律取下一個 value 的問題（不存在的未來、程式會出錯）已補充 |
| 小測 4 | 不能下結論：正確；已補其他 seed 都學會、seed 2 約 3000 個 episode 後恢復 |
| 小測 5 | delta 對應 δ、終止時去掉 V(s')：正確；λ=0.95 的理由補充：MC 在上千步 episode 太吵、rollout 固定長度需用 critic 補尾端、純 TD 會被落後的 critic 帶偏 |

## 待複習（非阻塞）

1. δ = target − V(s)：終止時 target 是 r，但 δ 還要減 V(s)。
2. 代數字前先確認每個 V 用的是哪個值（小測 1 把 V(B) 代成 0）。
3. critic 是 actor 看見未來的唯一管道；critic 落後時 TD advantage 會偏，且被壓低的 action 很少被走到，恢復很慢。
4. TD 與 MC 的 credit assignment：TD 把責任分給犯錯的那一步，MC 把後面所有運氣混進前面。

## 下一課入口（L09）

PPO：ratio 與 clip、GAE 的 λ（λ=0 是 TD、λ=1 是 MC）、value loss，對照 rsl_rl 原始碼與 Isaac Lab 訓練設定。
