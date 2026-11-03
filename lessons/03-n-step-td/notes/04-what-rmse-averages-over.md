# 第03课：一次RMSE在对什么取mean？

日期：2026-10-27。中文解释，使用English technical terms。

## 本次作答反馈

1. 同一条trajectory和同一组observed rewards下，改变alpha不会改变MC target，判断正确。更完整地说，target由sampled rewards与discount rate决定，learning rate只参与update；不能理解为target只依赖gamma。
2. old v(A)=1、alpha=0.1、MC target=1.458时，new estimate=1-0.1*(1-1.458)=1.0458，计算正确。
3. 若关心持续保持较低prediction error，应选method Y：其末50 episodes平均RMSE为0.10，低于X的0.20。X只有最后一次RMSE更低（0.01比0.05）。用户重视窗口平均、注意最后一次可能偶然的理由方向正确，但选择的字母与给定数据不一致。

## RMSE可以在不同维度上计算

RMSE要求对一组prediction errors取平方、mean和平方根。这组errors可以来自不同states或data points，并不一定来自不同episodes。“一次RMSE”表示一次evaluation，并不表示只使用一个error。

在本实验，一个episode结束时同时得到A、B、C、D四个value estimates；将它们分别与各自的true state values比较：

$$RMSE_e=\sqrt{\frac{1}{4}\sum_{s\in\{A,B,C,D\}}(\hat v_e(s)-v_\pi(s))^2}.$$

这里e标记episode结束时的evaluation，mean在四个states上进行。被比较的是value estimate与true value，并不是最后一次reward与某个预测。reward会影响update后的estimate，但不能直接替代true value。

## 两个states的简化例子

假设两个states的true values都是1：

| episode结束 | A estimate | B estimate | 本次RMSE |
|---|---:|---:|---:|
| 1 | 0 | 2 | sqrt((1+1)/2)=1 |
| 2 | 1 | 1 | 0 |

每次RMSE在两个states上汇总；这两个episodes的平均RMSE才是(1+0)/2=0.5。

## 本实验三个层次

1. 每个episode：对4个states计算一个RMSE。
2. 每个seed：对最后50个episodes的50个RMSE取mean。
3. 跨5个seeds：对5个window means做描述性summary。

由于每个episode都完成一次evaluation，一次300-episode run会产生300个RMSE值，再取最后50个做mean。如果evaluation频率改变，RMSE值的个数也会改变，不能把episode数和evaluation数普遍等同。

## Average accuracy与stability

较低mean RMSE支持较低的平均prediction error，不直接证明更stable。要比较fluctuation，还要检查error curve、跨episodes的standard deviation等；本实验表中的seed间standard deviation又是在描述不同runs的差异，不能代替单条curve的时间波动。

## 快速迁移检查

假设一次evaluation评估10个states，训练100个episodes且每个episode结束都做一次evaluation：

1. 每次RMSE对多少个prediction errors取mean？
2. 整个run会得到多少个RMSE值？

这两项用于确认统计维度，不要求写代码或运行网页。


## 最新比较方向澄清

| metric | X | Y | 此metric下较低error |
|---|---:|---:|---|
| 最后一次evaluation的RMSE | 0.01 | 0.05 | X |
| 末50 episodes平均RMSE | 0.20 | 0.10 | Y |

准确表述：Y在这50 episodes中的平均prediction error较低；X在最后一次evaluation的prediction error较低。仅凭这两个summary，无法判断哪条curve逐渐稳定，需要查看曲线或时间上的fluctuation指标。
