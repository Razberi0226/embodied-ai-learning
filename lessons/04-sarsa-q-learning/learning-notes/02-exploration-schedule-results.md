# 2026-10-28：exploration schedule核心实验结果与反馈

来源：导师此前已本地运行并验证的参考数据，本轮读取既有结果，未重跑网页。用户已经提交预测、代码概念解释与访问概率计算；没有声称用户已独立运行。扩展WARM_START=True的结果暂不展示，先提交扩展预测。

## 第3、4项批改

- fast的有限预算coverage风险预测正确。机制应涵盖可能遗漏good和进入B的机会；避免bad本身不一定是失败，缺乏判断actions价值的经验才是关键。
- epsilon_at(e,schedule)由episode编号与预先指定的schedule规则计算；用户的“根据策略和episode”方向正确，这里的schedule具体指fixed/linear/fast规则，不是action采样频率，也不是N(s,a)。
- q=0、N=0不能证明action不好，用户解释正确。N=0表示没有该pair的直接样本。
- 访问(B,bad)的两次计算0.1*0.1=0.01与0.9*0.1=0.09均正确；epsilon相同而greedy ranking不同，会改变实际访问分布。

## 配置与结果

固定Q-learning、gamma=0.9、q初始化全0、alpha=1/N。每个seed每个schedule训练2000 episodes，5个seeds为7、19、42、101、2026。WARM_START=False。training指标是末500个episodes的discounted return平均，再跨seeds平均；greedy evaluation冻结q，不explore。

| schedule | 最终greedy走shortcut的runs | greedy return均值 | training末窗口均值 | 平均environment steps |
|---|---:|---:|---:|---:|
| fixed | 5/5 | 1.800000 | 0.752320 | 3775.8 |
| linear | 5/5 | 1.800000 | 1.694240 | 3883.4 |
| fast | 2/5 | 1.320000 | 1.320000 | 2749.0 |

相同episodes不等于相同environment steps，不能用这个表直接声称固定transition预算的学习效率。5个seeds是当前配置的描述性结果，不是任意任务上的排序或成功概率证明。

## fast的每个seed

| seed | N(S,shortcut) | N(B,good) | N(B,bad) | q(S,shortcut) | 最终greedy路线 |
|---|---:|---:|---:|---:|---|
| 7 | 0 | 0 | 0 | 0.000000 | safe |
| 19 | 3 | 2 | 1 | 0.600000 | safe |
| 42 | 1742 | 1742 | 0 | 1.798967 | shortcut -> good |
| 101 | 2000 | 2000 | 0 | 1.799100 | shortcut -> good |
| 2026 | 0 | 0 | 0 | 0.000000 | safe |

seed 7与2026完全没选过shortcut，因此B中两个q=0仍是初始化。它们不是实际试出shortcut只有0收益。

seed 19选过shortcut三次，good两次、bad一次；B中的terminal estimates已经是2和-10，但q(S,shortcut)仍只有0.6，低于safe的1。这里还涉及one-step bootstrapping的信息传播：早期更新S时，B的estimates可能尚未学好；后续缺少再次访问S的shortcut的机会，旧偏差未充分修正。因此N>0也不等于estimate已经准确。

fast也有2个seeds找到shortcut，不能说它必然失败。

## 核心实验全部seed统计

| seed | schedule | N(S,safe) | N(S,shortcut) | N(B,good) | N(B,bad) | greedy return | steps |
|---|---|---:|---:|---:|---:|---:|---:|
| 7 | fixed | 242 | 1758 | 1570 | 188 | 1.800 | 3758 |
| 7 | linear | 143 | 1857 | 1767 | 90 | 1.800 | 3857 |
| 7 | fast | 2000 | 0 | 0 | 0 | 1.000 | 2000 |
| 19 | fixed | 231 | 1769 | 1584 | 185 | 1.800 | 3769 |
| 19 | linear | 120 | 1880 | 1808 | 72 | 1.800 | 3880 |
| 19 | fast | 1997 | 3 | 2 | 1 | 1.000 | 2003 |
| 42 | fixed | 218 | 1782 | 1605 | 177 | 1.800 | 3782 |
| 42 | linear | 104 | 1896 | 1806 | 90 | 1.800 | 3896 |
| 42 | fast | 258 | 1742 | 1742 | 0 | 1.800 | 3742 |
| 101 | fixed | 191 | 1809 | 1637 | 172 | 1.800 | 3809 |
| 101 | linear | 88 | 1912 | 1826 | 86 | 1.800 | 3912 |
| 101 | fast | 0 | 2000 | 2000 | 0 | 1.800 | 4000 |
| 2026 | fixed | 239 | 1761 | 1584 | 177 | 1.800 | 3761 |
| 2026 | linear | 128 | 1872 | 1793 | 79 | 1.800 | 3872 |
| 2026 | fast | 2000 | 0 | 0 | 0 | 1.000 | 2000 |

## 下一步：warm-start单变量扩展

只改WARM_START=False为True。最初4个episodes指定：safe；shortcut后good；shortcut后bad；shortcut后good。它们计入同一个2000 episodes预算，q仍从全0开始，按实际reward照常update；target、alpha规则、seeds和epsilon函数均不变，之后episode编号不重置。

第5项先提交预测：这会怎样影响fast找到shortcut的机会？为什么？改变的是epsilon函数，还是最初的经验采集安排？是否增加了总episode预算？看结果后再解释证据适用范围。

第6项同步做第03课RMSE短复查：每次evaluation评估10个states，每个episode末evaluation一次，共100个episodes；每个RMSE平均多少个平方误差？整段有多少个RMSE数？末段平均RMSE较低，能否单独证明波动更小？

无需从零代码或立即运行网页。扩展预测与第6项集中提交后，再展示已验证扩展结果并据实际回答判断收尾。
