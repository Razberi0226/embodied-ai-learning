# 第 03 课：n-step TD、MC 与公平比较

接着[第 02 课](../02-mc/README.md)。核心问题：一步 TD 和完整回报的 MC 之间还有哪些选择？比较两种方法时，怎样才算公平？

学习日期：2026-10-26 至 2026-10-28。每天计划约 120 分钟，含休息。

## 学习顺序

| 顺序 | 材料 | 内容 |
|---|---|---|
| 1 | [二步 TD 先学小节](notes/01-two-step-td-intro.md) | 从一步到二步：目标与更新分开写；提前终止时只用已经发生的 reward |
| 2 | [n-step TD 与 MC 衔接](notes/02-n-step-td-and-mc.md) | 一般 n-step target；第 n 步前就终止的处理；MC 是“一直看到 terminal”的端点 |
| 3 | [完整 trajectory 与公平比较](notes/03-full-trajectory-fair-comparison.md) | 把 n-step target 用到整段 trajectory；速度、波动和公平比较；确定与随机 reward 两个实验 |
| 4 | [RMSE 在对什么取 mean](notes/04-what-rmse-averages-over.md) | 先对 states、再对多次 evaluation 汇总；平均误差低不等于波动小 |
| — | [提交模板](notes/submission-template.md) | 第 3 份讲义的作答格式 |

## 实验

| 脚本 | 说明 |
|---|---|
| [two_step_target_exercise.py](experiments/two_step_target_exercise.py) | 练习：补全二步 target 函数；未补全时只打印提示 |
| [n_step_target_exercise.py](experiments/n_step_target_exercise.py) | 练习：补全一般 n-step target 函数 |
| [n_step_target_checked.py](experiments/n_step_target_checked.py) | 已检查的参考实现，四组输入：5.436、2.52、4.0、1.5 |
| [full_trajectory_comparison.py](experiments/full_trajectory_comparison.py) | 练习版：需补全 `target_from_trajectory` |
| [full_trajectory_comparison_runnable.py](experiments/full_trajectory_comparison_runnable.py) | 可运行版。默认 `MODE='check'`，再依次改为 `'deterministic'`、`'stochastic'` |

```bash
python lessons/03-n-step-td/experiments/n_step_target_checked.py
python lessons/03-n-step-td/experiments/full_trajectory_comparison_runnable.py
```

## 本课要点

- n-step target 把**已经收到的 reward 全部折扣累加**，未终止时再加上 \(\gamma^n\) 乘以第 n 步之后 state 的 value estimate。真正终止时只取消这一项后续估计，不丢掉已收到的 reward。
- MC 可以看成 n-step 框架中“一直看到 terminal”的端点，但它和 TD(0) 的更新目标、更新时机仍然不同。
- 教学实验在 episode 结束后，用 episode 开始时冻结的 estimates 统一更新，这是本课的实验设定，不是唯一实现。
- 公平比较要固定其他条件：相同 trajectories 或相同随机 returns、相同预算、多个 seeds，并说明误差是在哪个维度上汇总的。

## 学习记录

- [二步 TD 学习记录](learning-notes/01-two-step-td.md)
- [n-step 函数排错记录](learning-notes/02-n-step-function-debugging.md)
- [learning rate 对照与 stochastic reward](learning-notes/03-learning-rate-and-stochastic-reward.md)
- [MC target 与 stochastic 比较](learning-notes/04-mc-target-and-stochastic-comparison.md)
- 第 03、04 课的收尾记录在[第 04 课](../04-sarsa-q-learning/learning-notes/03-lessons-03-04-wrap-up.md)

## 教材对应

赵世钰《强化学习的数学原理》第 7 章（Temporal-Difference Methods）。数值例子和实验为本课自编的教学设置。
