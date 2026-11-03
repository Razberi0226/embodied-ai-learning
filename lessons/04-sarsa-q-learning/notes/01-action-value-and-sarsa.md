# 第04课先学：从state value到action value，再到Sarsa

发布日期：2026-10-27。根据“好的我懂了，继续吧”的要求，在第03课后继续一个20—30分钟的衔接小节。预计时间不是实际学习时长。本节不把第03课RMSE问题的“懂了”记成独立验证通过；统计维度与metric解释保留短复查。

## 核心问题

前面固定trajectory中没有比较actions：A只能到B，B只能到C。我们主要做policy evaluation。现在开始比较同一个state里的不同actions，为policy improvement打基础。

只知道一个state的state value，还不能直接知道其中哪个action更值得先尝试。action value把“第一步选哪个action”单独列出来。

## state value与action value

- v_pi(s)：从state s出发，第一步及之后都按policy pi选择actions时的expected return。
- q_pi(s,a)：从state s出发，第一步先指定action a，之后按policy pi行动时的expected return。

q_pi同样考虑整个后续return，不是单次reward。

它们满足：

$$v_\pi(s)=\sum_a\pi(a\mid s)q_\pi(s,a).$$

因此state value是当前policy对不同action values加权后的平均；不是默认取最大的action value。

## 一个机器人例子

state s：夹爪已接近物体。actions为close（直接闭合）和align（先对准）。假设在某个固定policy pi下，真实action values已知：

| action | q_pi(s,a) | pi(a\|s) |
|---|---:|---:|
| close | 2 | 0.25 |
| align | 6 | 0.75 |

于是v_pi(s)=0.25*2+0.75*6=5。

5描述当前policy随机选择第一步action时的expected return；如果第一步指定align、随后仍按pi，expected return是6。这些数字不是“close立即拿2分、align立即拿6分”。

若根据这组action values进行greedy action selection，当前会选align。怎样保留exploration、怎样持续进行policy improvement，后面再具体展开。

## Sarsa：用TD学习action values

前面的state-value TD target为r_{t+1}+gamma*v_t(s_{t+1})。Sarsa使用实际选中的next action，构造：

$$y_t=r_{t+1}+\gamma q_t(s_{t+1},a_{t+1}).$$

沿用赵老师教材的TD error符号方向：

$$\delta_t=q_t(s_t,a_t)-y_t,$$
$$q_{t+1}(s_t,a_t)=q_t(s_t,a_t)-\alpha\delta_t.$$

这里q_pi是给定policy的真实action value；q_t是当前estimate。每次只更新实际访问的state–action pair；其他pairs的estimate不因这一次update自动改变。

Sarsa名称来自每次update所需的五项：

```text
State, Action, Reward, next State, next Action
s_t,   a_t,    r_(t+1), s_(t+1),    a_(t+1)
```

关键是next action已经由正在遵循的policy实际选定，target读取这个action对应的q estimate。当前先理解固定policy的action-value evaluation；加入policy improvement后，才进一步成为control学习过程。

如果next state真正terminal，target只有本次reward，不再要求next action，也不再添加terminal之后的q estimate。

## 完整算例

current q estimate=3，observed reward=1，discount rate=0.9，实际next action的q estimate=4，learning rate=0.2：

```text
TD target = 1 + 0.9*4 = 4.6
TD error  = 3 - 4.6 = -1.6
new q     = 3 - 0.2*(-1.6) = 3.32
```

若这次next state真正terminal，则target=1，TD error=2，new q=2.6。

## 可运行演示与检查范围

同目录“sarsa_single_step.py”包含以上action-value加权和两种单次update，代码已由导师在本地检查。它不是完整environment训练，也没有训练真实机器人或执行policy improvement。用户不需要从零写代码，不要求立即操作网页运行。

## 一组练习：三项集中提交

1. 某state有两个actions L、R，q_pi(s,L)=1、q_pi(s,R)=5；pi(L|s)=0.4、pi(R|s)=0.6。求v_pi(s)；根据这些action values进行greedy action selection时会选哪个action？
2. current q_t(s,a)=2，reward=0，gamma=0.9，alpha=0.2。next state的两个action estimates为q_t(s_next,L)=4、q_t(s_next,R)=1；policy这次实际选中R。求Sarsa的TD target与更新后的q_t(s,a)。
3. 如果第2题改为next state真正terminal，reward仍为0，那么TD target是多少？是否还需要next action？

提交格式：第1题“state value、action”，第2题“target、new estimate”，第3题“target、是否需要next action”。不要求代码实现或重新运行网页。

## 教材对应

- 赵世钰教材第2章state value与action value的关系，第7章7.2节“TD learning of action values: Sarsa”，式(7.12)。
- 官方教材主页：https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning
- 第7章：https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/3%20-%20Chapter%207%20Temporal-Difference%20Methods.pdf

机器人情境与数字是本小节的教学例子，不是教材实验结果。完整Sarsa、exploration与policy improvement将在理解本组目标之后继续。
