# 第03、04课收尾记录

确认日期：2026-10-28。完成指当前阶段核心学习目标通过；已反馈的措辞和概念精度问题继续间隔复习。计划每日约120分钟，用户未报告实际学习时长，不将计划量记为实际用时。

## 本次提交与反馈

用户第5项：初始经验提供有效q estimates，帮助后续action选择，提高fast找到shortcut的机会；一般任务需要多少经验应根据任务判断。预测和因果解释正确，也正确限制了结论范围。预算与epsilon函数保持不变是题目给定设置，未额外声称用户独立解释过所有实现细节。

用户第6项：每次RMSE平均10个平方误差，100个episodes产生100个RMSE，较低末段mean不能单独证明较低波动。三项均正确，完成第03课留下的统计维度复查。

一次evaluation的公式为RMSE=sqrt(sum_{i=1}^{10}(estimate_i-true_value_i)^2/10)。这里的平均在states上；100次evaluation各产生一个RMSE。之后还可以沿evaluations再求mean；mean error与波动是不同统计属性。用户没有提供波动数值，不编造std或曲线分析结果。

## warm-start扩展的导师参考结果

两组设置都是Q-learning、q全0初始化、gamma=0.9、alpha=1/N、2000 episodes、seeds 7/19/42/101/2026。扩展只改变初始四个episodes的采样安排，仍计入同一总预算；之后epsilon函数和episode编号不变。

| schedule | 无warm start找到shortcut | 有warm start找到shortcut | 无warm start的greedy return均值 | 有warm start的greedy return均值 |
|---|---:|---:|---:|---:|
| fixed | 5/5 | 5/5 | 1.800000 | 1.800000 |
| linear | 5/5 | 5/5 | 1.800000 | 1.800000 |
| fast | 2/5 | 5/5 | 1.320000 | 1.800000 |

这支持本小确定MDP下初始经验影响后续estimates和behavior的解释。四次安排不等于一般随机机器人任务的充分样本量。相同episode预算也不等于相同environment-step预算，不从本表推断固定transition预算的效率排序。

## fast扩展各seed统计

| warm start | seed | N(S,shortcut) | N(B,good) | N(B,bad) | q(S,shortcut) | greedy return | steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| False | 7 | 0 | 0 | 0 | 0.000000 | 1.000000 | 2000 |
| False | 19 | 3 | 2 | 1 | 0.600000 | 1.000000 | 2003 |
| False | 42 | 1742 | 1742 | 0 | 1.798967 | 1.800000 | 3742 |
| False | 101 | 2000 | 2000 | 0 | 1.799100 | 1.800000 | 4000 |
| False | 2026 | 0 | 0 | 0 | 0.000000 | 1.000000 | 2000 |
| True | 7 | 1998 | 1997 | 1 | 1.799099 | 1.800000 | 3998 |
| True | 19 | 1998 | 1997 | 1 | 1.799099 | 1.800000 | 3998 |
| True | 42 | 1997 | 1996 | 1 | 1.799099 | 1.800000 | 3997 |
| True | 101 | 1999 | 1998 | 1 | 1.799100 | 1.800000 | 3999 |
| True | 2026 | 1999 | 1997 | 2 | 1.799100 | 1.800000 | 3999 |

## 第03课完成范围

- 二步及n-step targets、提前terminal的边界、完整observed return与MC的联系。
- frozen-estimate教学比较与普通逐transition update的设定差别。
- learning rate、首次非零传播与误差达标速度的区别。
- stochastic reward的sample与expectation，以及RMSE统计轴和结论范围。

依据既有手算、代码语句解释、参数对照预测和本次RMSE独立作答确认核心目标。trajectory前向方向与value信息传播方向的措辞继续复习。

## 第04课完成范围

- Sarsa、Expected Sarsa、Q-learning的targets与terminal处理。
- Expected Sarsa只对已到达state下的next actions求expectation；区分environment随机性。
- behavior policy、target policy、on/off-policy和training/evaluation的区别。
- epsilon-greedy、schedule、state-action coverage、N=0的含义与两阶段访问概率。
- 固定条件的核心实验预测、单变量warm-start预测，以及不把小环境结果推广为普遍结论。

依据用户实际计算、关键代码解释、运行前预测、因果解释和范围判断确认核心目标。导师提供并运行的代码已标明来源；不声称用户从零实现、亲自在Programiz运行或操作过远程GPU。

## 后续间隔复习，不是待交作业

1. greedy/non-greedy的0.9/0.1标签及exploration分支仍可能选中greedy。
2. action概率与样本频率；固定epsilon时q ranking变化才可能改变此实现的action分配。
3. 同一state指定action后，后续policy改变会改变action value；behavior return与optimal continuation区别。
4. experience沿trajectory向前，value信息可通过bootstrapping传回较早pairs。
5. N>0不自动等于estimate准确；episode与environment-step预算、均值与波动的区分。

## 下一步

本次收尾，不追加新题。下一正式包拟为第05课：Q-table的局限、features与function approximation，从最小线性模型开始，为后续DQN做准备。按用户反馈调整，不要求今天预习或从零写代码。
