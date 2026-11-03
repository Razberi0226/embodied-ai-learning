# 第05课学习记录：多个samples、learning rate、coverage 与 TD target 接口

日期：2026-11-03（2026-10-30 发布的学习包，本日完成作答与验收）
结果来源：学习者在浏览器内的 Python 环境亲自运行，输出与 2026-10-30 导师参考输出逐项一致；B、C 两项修改另由导师用电脑版 Python 复核。
实际用时：未计时。

## 结论：第05课完成

| 部分 | 结果 |
|---|---|
| 小测1 batch gradient 计算 | 最终 weights (1.2, 0.1) 正确；error 写成 target−prediction，gradient 写成 (2,1)，实为 −∇J。已讲按本课定义 e2=−2、∇J=(−2,−1) |
| 小测2 边算边改 | 核心正确；补充“不保证相同”：按 (0,0)→(2,4) 顺序恰好与 batch 相同，反序 w0=0.095 |
| alpha 预测 | 0.2 更接近、2.2 不会更快：正确；“在真值附近震荡”只对一半，w0 实际发散 |
| 代码阅读 | 取平均的行正确；“同一位置”的理由已补充为代码证据（循环内无 weights 赋值，return 同时由旧值算出） |
| coverage 预测 | w1 wide 更近：正确；共同 eval RMSE 预测反了（narrow 2.32，wide 0.008），已纠正 |
| A 推导 | w0 乘数 −1.2 正确；w1 乘数误用 narrow 的 c，wide 应为 1−2.2×2/3≈−0.467；稳定上限 α<2 正确 |
| B STEPS=80 | 0.05 接近真值、0.2 基本不变：正确；记录 2.2 组 updates=75、sample_uses=225、parameter_guard_stop |
| C NARROW=(−0.2,0,0.2) | 识别 c 变 4 倍；数值预测经讲解后理解：w1≈0.385，三个 eval 点全部为 extrapolation |
| 小测3 | 结论正确，alpha 不改变 targets 正确；变量 w1/w0 标签笔误；应引用实验数值作证据 |
| 小测4 | 方向正确（比较 error），已精确为“同一组 eval 位置、对可靠 reference 的 error”；无法自行举出证据范围外的结论，已给例子，待下次自拟一条 |
| 小测5 semi-gradient TD | 经换数字例题讲解后独立完成：target 1.8、error −0.8、gradient (−0.8,−0.8)、w=(1.08,0.08)；终止时 target 0、w=(0.9,−0.1)；检查 prediction 更接近 target。全部正确 |

## 关键数据（用户运行）

- alpha 对照，40 次：0.05 → (1.485, 0.871)，eval RMSE 0.709；0.2 → (1.993, 0.99987)，0.008；2.2 → w1=2、w0=−1468.77，train loss ≈1.08×10⁶。
- coverage，alpha=0.2，40 次：wide w1=1.993、eval RMSE 0.008；narrow(±0.1) w1=0.104、train loss 0.012、eval RMSE 2.32。
- 闭式检验：x 平均为 0 时，(w0−1) 每步乘 (1−α)，(w1−2) 每步乘 (1−αc)，c=x² 的平均。±0.1 与 ±0.2 的 w1 预测（0.104、0.385）与运行结果一致。

## 待复习（非阻塞）

1. error 符号约定：prediction − target；gradient 与 −gradient（下降方向）不要混用。
2. train loss 与共同 eval error 的区别；比较时写明 update 次数与计算位置。
3. 证据范围：下次开场用自己的话写一条本实验无法证明的结论。
4. 同类题注意题目指定的数据集（wide/narrow 的 c 不同）。

## 下一课入口（L06）

已提到但未考查：更新 w 后 target 本身会变（moving target）。例：第5题更新后，用新 w 重算 target 为 0.9×2.24=2.016，差距从 0.8 变为 0.856。L06 从 semi-gradient TD(0) + linear function approximation 开始，再到 action-value 近似与 DQN 中 target network 的动机。
