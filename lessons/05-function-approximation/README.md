# 第 05 课：从 table 到 function——value function approximation 入门

核心问题：state 多到无法逐个存表时，用一组共享 parameters 表示 value。这样做带来了什么好处，又带来了什么新问题？

学习日期：2026-10-28（预习）、2026-10-30 发布正式学习包，2026-11-03 完成。计划约 120 分钟，含休息。

## 学习顺序

| 顺序 | 材料 | 内容 |
|---|---|---|
| 1 | [从 table 到 function](notes/01-from-table-to-function.md) | 线性表示 \(\hat v(x,w)=w_1x+w_0\)；改共享 parameter 会同时影响其他 states；没见过的输入有输出，不等于知道真实 value |
| 2 | [一次 gradient step](notes/02-one-gradient-step.md) | half squared loss；gradient \((e\cdot x,\ e)\)，其中 \(e=\text{prediction}-\text{target}\)；\(w\leftarrow w-\alpha\nabla\) |
| 3 | [多个 samples 与 evaluation](notes/03-multiple-samples-and-evaluation.md) | batch gradient；learning rate 对照；sampling coverage；train loss 与共同 eval error；接到 TD target |

## 实验

| 脚本 | 说明 |
|---|---|
| [shared_parameters_demo.py](experiments/shared_parameters_demo.py) | table 与共享 parameters 的对照，不含训练 |
| [fixed_target_update.py](experiments/fixed_target_update.py) | 固定 target 的一次 gradient step |
| [multiple_samples_experiment.py](experiments/multiple_samples_experiment.py) | 核心实验只改 alpha，扩展实验只改训练位置；[参考输出](experiments/multiple_samples_reference_output.txt) |
| [l06_preview_moving_target.py](experiments/l06_preview_moving_target.py) | 第 06 课预习：共享 parameters 下，semi-gradient TD 的 target 会跟着 w 变 |

```bash
python lessons/05-function-approximation/experiments/multiple_samples_experiment.py
python lessons/05-function-approximation/experiments/l06_preview_moving_target.py
```

## 本课要点

- batch gradient 要用同一组旧 weights 算完所有 samples 的平均 gradient，再统一更新一次。
- 本例训练位置的 x 平均为 0，两个 parameters 的更新互不影响：\((w_0-1)\) 每步乘以 \(1-\alpha\)，\((w_1-2)\) 每步乘以 \(1-\alpha c\)，其中 \(c\) 是 \(x^2\) 的平均。乘数的绝对值大于 1 就发散，所以本例需要 \(\alpha<2\)。
- feature 尺度小，slope 学得慢：narrow 训练位置下 40 次后 \(w_1\) 只到约 0.10，train loss 却不大。比较时要在**同一组 eval 位置**上，对可靠的 reference 计算 error，而不是比较各自的 train loss。
- semi-gradient TD：先用旧 w 算出 \(r+\gamma\hat v(x_{next},w)\)，这一步把它当常数做 gradient step；真正终止时 target 只有 r。w 改变后，下一次的 target 也会变（moving target），这是第 06 课的起点。

## 学习记录

[第 05 课学习记录](learning-notes.md)

## 教材对应

赵世钰《强化学习的数学原理》[第 8 章 Value Function Methods](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)。batch 与逐 sample 更新可对照 [Stanford CS229 讲义](https://cs229.stanford.edu/notes-spring2019/cs229-notes1.pdf)。数值实验为本课自编的教学设置。
