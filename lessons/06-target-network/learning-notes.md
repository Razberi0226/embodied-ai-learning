# 第06課學習記錄：更新分佈與 target network

日期：2026-11-04（當天發布、當天完成）
結果來源：學習者在瀏覽器內的 Python 環境運行，輸出與導師參考輸出逐項一致；P_T1=100、SYNC=100 一組由導師運行提供。
實際用時：未計時。

## 結論：第06課完成

| 部分 | 結果 |
|---|---|
| 閉卷回憶 | 負號、1−α 正確；error 寫成 old−new，已更正為 prediction−target；train loss 誤說成「衡量梯度」，已更正；證據範圍自擬一條：第05課 target 固定，證明不了 target 會動時的收斂，接受 |
| 模組 A | 初版太密，學習者表示完全看不懂；改用「乘數」主線的 8 步重講（已附講義末尾）後，能用 trace/det 正確判斷 P_T1=80 收斂且擺動、95 發散 |
| 核心實驗 | 三組結果與判斷一致；觀察到 td_loss 降得比 true_error 快（正確，平方量且相對 target）；兩處程式定位正確；P_T1=90 時 trace 算成正值（應為 −0.13），分析本應預測發散，實際 α=0.1 固定順序下慢慢收斂；假設（平均化、α 很小）由導師說明 |
| 擴充實驗 | SYNC=1 等於沒有 target network、SYNC 太大收斂慢、沒有 T2 時 target network 也救不了：均正確；SYNC 取捨推理正確 |
| 小測 1 | 正確：0<p<0.877，p=0.6 收斂 |
| 小測 2 | 方向正確；已補精確版：T1 表 trace −0.7、T2 表 trace 5，兩表 det 各為 0，合起來 p(1−p) |
| 小測 3 | 起初繞住（沒分清凍結的 v̂(2)=2 與學到的 v̂(2)）；講解後答案 v̂(1)=1.8、v̂(2)=1，穩定但錯誤 |
| 小測 4 | 結論「不能」正確，理由（沒有 target network）錯；同一實驗 P_T1=95 即反例 |
| 小測 5 | sync 對應 epochs×mini-batches 正確；「資料比例」一詞不認得，已說明＝更新分佈，PPO 接近 P_T1=50；value loss 低不代表準：正確，已補數字與 bootstrapped target 的原因 |
| 短檢查（w_T 停在 (0,0)） | T1 target=0 正確；把 v̂(2) 也答成 0，應為 1（T2 終止，target 固定是 r=1，與 target network 無關）。正確答案 w=(1,−1)、v̂(1)=0、v̂(2)=1 |

## 待複習（非阻塞）

1. 終止 transition 的 target 只有 r，不受 target network 影響（短檢查漏掉）。
2. 證據範圍：先在同一個實驗裡找反例，再談推廣。
3. 分析的假設：平均化與 α 很小；靠近臨界值時要實際驗證。
4. 名詞：update distribution（更新分佈、資料比例）。

## 下一課入口（L07）

policy gradient（REINFORCE，教材第 9 章）：從「學 value」轉到「直接學 policy」，為 L08 actor-critic 做準備。
