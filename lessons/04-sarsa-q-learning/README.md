# 第 04 课：Sarsa、Expected Sarsa、Q-learning 与 exploration

核心问题：从评估 state value 走向学习选择 actions 时，TD target 里的 next action 怎么处理？exploration 和数据覆盖又怎样改变学到的行为？

学习日期：2026-10-27 至 2026-10-28。每天计划约 120 分钟，含休息。原计划表中的 04—06 三个主题已合并进本课。

## 学习顺序

| 顺序 | 材料 | 内容 |
|---|---|---|
| 1 | [action value 与 Sarsa](notes/01-action-value-and-sarsa.md) | 从 \(v_\pi(s)\) 到 \(q_\pi(s,a)\)；Sarsa 用实际选到的 next action 构造 target |
| 2 | [MC 的边界与三种 TD target](notes/02-three-td-targets.md) | Sarsa、Expected Sarsa、Q-learning 的 target 对照；behavior policy 与 target policy |
| 3 | [Expected Sarsa 平均了哪部分随机性](notes/03-expected-sarsa-randomness.md) | 对 next action 求 expectation，不对 environment 的随机性求平均 |
| 4 | [on-policy 与 off-policy](notes/04-on-policy-off-policy.md) | 两步 MDP 上比较 Sarsa 与 Q-learning；training return 与冻结后的 greedy evaluation |
| 5 | [exploration schedule 与 state-action coverage](notes/05-exploration-schedule-and-coverage.md) | fixed、linear、fast 三种 ε 安排；没访问过的 state-action 只有初始值 |

## 实验

| 脚本 | 说明 |
|---|---|
| [sarsa_single_step.py](experiments/sarsa_single_step.py) | 一次 Sarsa update 的演示，含终止情形 |
| [three_td_targets.py](experiments/three_td_targets.py) | 同一组输入下三种 TD target 的数值 |
| [on_off_policy_two_step.py](experiments/on_off_policy_two_step.py) | 两步 MDP：Sarsa 与 Q-learning，5 个 seeds；扩展只改 `EPSILON` |
| [exploration_coverage.py](experiments/exploration_coverage.py) | Q-learning 在三种 ε 安排下的路线与访问次数；扩展只改 `WARM_START` |

```bash
python lessons/04-sarsa-q-learning/experiments/three_td_targets.py
python lessons/04-sarsa-q-learning/experiments/on_off_policy_two_step.py
python lessons/04-sarsa-q-learning/experiments/exploration_coverage.py
```

## 本课要点

- state value 是按 policy 对 action values 加权：\(v_\pi(s)=\sum_a \pi(a|s)\,q_\pi(s,a)\)。
- 三种 target 的差别在 next action：Sarsa 用实际选到的，Expected Sarsa 按 policy 取 expectation，Q-learning 取 max。
- 判断 on-policy 还是 off-policy，看 behavior policy 与 target policy 是否允许不同，而不是看有没有 exploration。
- 带 exploration 的 training return 和冻结 q 之后的 greedy evaluation 回答的是不同问题。例如在本课的两步 MDP 中，ε=0.2 时 Sarsa 的 greedy 路线是 safe（return 1.0），Q-learning 走 shortcut（1.8）。
- 在同样 2000 个 episodes 下，ε 衰减太快的 fast 安排只有 2/5 个 seeds 找到 shortcut；开头强制尝试四次（warm start）后变成 5/5。这些是该小型确定 MDP 上的结果，不代表一般任务需要的经验量。

## 学习记录

- [on/off-policy 实验结果与反馈](learning-notes/01-on-off-policy-results.md)
- [exploration schedule 实验结果与反馈](learning-notes/02-exploration-schedule-results.md)
- [第 03、04 课收尾记录](learning-notes/03-lessons-03-04-wrap-up.md)

## 教材对应

赵世钰《强化学习的数学原理》第 7 章（Sarsa、Expected Sarsa、n-step Sarsa、Q-learning，以及 on-policy 与 off-policy）。两步 MDP 为本课自编的教学环境。
