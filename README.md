# 具身智能学习笔记

从强化学习基础开始，通过讲解、手算和可复现实验，逐步理解具身智能。

## 课程目录

| 课次 | 主题 | 内容 |
|---|---|---|
| [第 01 课](lessons/01-td/README.md) | TD(0) 状态价值预测 | 奖励与回报、TD 目标、更新顺序、学习率、折扣、终止状态 |
| [第 02 课](lessons/02-mc/README.md) | MC、增量平均与固定学习率 | 实际完整回报、MC 与 TD、1/n 更新、随机回报与公平对照 |

## 第 02 课怎么学

1. 阅读[讲义](lessons/02-mc/README.md)，区分实际回报、期望价值和估计。
2. 完成[习题](lessons/02-mc/exercises.md)，再对照[参考解答](lessons/02-mc/answers.md)。
3. 运行[MC 与 TD 对照实验](lessons/02-mc/experiments/mc_td_comparison.py)和[增量均值实验](lessons/02-mc/experiments/incremental_mean.py)，保留预测与输出。
4. 阅读[学习记录](lessons/02-mc/learning-notes.md)，复习目标选择、样本数与波动来源。

```bash
python lessons/02-mc/experiments/mc_td_comparison.py
python lessons/02-mc/experiments/incremental_mean.py
```

## 第 01 课怎么学

1. 阅读[讲义](lessons/01-td/README.md)，先建立概念。
2. 完成[手算和小测](lessons/01-td/exercises.md)。
3. 按顺序运行[实验](lessons/01-td/experiments/)，先预测，再观察输出。
4. 对照[参考解答](lessons/01-td/answers.md)和[学习记录](lessons/01-td/learning-notes.md)。

所有 `.py` 实验只用 Python 标准功能，无需安装第三方包或使用 GPU。可以在本地运行，也可以将任意一个脚本的完整内容复制到 [Programiz 在线 Python](https://www.programiz.com/python-programming/online-compiler/) 后运行。

在仓库根目录运行：

```bash
python lessons/01-td/experiments/two_state_td.py
python lessons/01-td/experiments/learning_rate_comparison.py
python lessons/01-td/experiments/discount_factor_comparison.py
python lessons/01-td/experiments/three_state_td.py
```

喜欢逐段运行时，可以打开 [Notebook](lessons/01-td/td_learning.ipynb)。Notebook 中保留了需要自己完成的练习，完整三状态实现见独立脚本。

## 学习约定

- 每次围绕一个核心问题，计划约 2 小时，包含休息。
- 运行前记录预测，运行后解释结果，保留设置和输出。
- 区分“程序能运行”“能够解释结果”和“结论可以推广”。
- 本课的固定路径用于理解算法，后续再扩展到动作选择、随机环境和具身任务。

## 学习资料

- [赵世钰课程与教材主页](https://shiyuzhao.westlake.edu.cn/Teaching.htm)
- [《强化学习的数学原理》教材、分章 PDF 和幻灯片](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)

本仓库记录个人学习过程，讲义与实验是配套学习材料。赵世钰原课程和教材请通过以上作者链接阅读。
