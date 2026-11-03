# 第04课补讲：Expected Sarsa对哪部分随机性求了平均？

日期：2026-10-27。针对用户已正确计算三种TD targets、但对剩余随机性尚不明确的反馈。不是新的完整日课，不要求编写或运行代码。

## 已提交内容的反馈

新例子q_next(L)=6、q_next(R)=-2，pi(L)=0.75、pi(R)=0.25，实际next action为R，reward=1、gamma=0.5。

- Sarsa target=1+0.5*(-2)=0，正确。
- policy加权estimate=0.75*6+0.25*(-2)=4，Expected Sarsa target=1+0.5*4=3，正确。
- Q-learning target=1+0.5*6=4，正确。
- 只改实际next action时，仅Sarsa target改变；Expected Sarsa不要求同时执行两种actions，正确。
- Q-learning允许behavior policy实际选R、target仍使用max对应L，判断正确。其target使用greedy规则，实际action可以用于exploration。
- Expected Sarsa未消除环境reward与state transition的随机性，判断正确；用户明确希望补讲原因，不将这句判断记为充分解释通过。

## 两层随机性

固定当前state-action pair (s,a)和当前q estimates。一次transition可分为：

1. environment给出本次reward r和next state s'。即使重复相同(s,a)，结果也可能不同。
2. 到达s'后，policy按pi选择next action a'。随机policy可能选出不同actions。

Sarsa target=r+gamma*q_est(s',a')，因此两层抽样都可能影响target。

Expected Sarsa target=r+gamma*sum_a pi(a|s')*q_est(s',a)。它读取这次实际observed reward和实际到达的next state，只将第二层的action抽样替换成该policy的加权平均。

pi通常是我们指定、能够计算概率的policy；model-free设置通常并不知道环境完整的reward/transition概率，因此仍使用环境本次给出的样本。

## 一个两阶段例子

重复同一个当前(s,a)，环境有50%概率到达U、50%概率到达D；两个next states都非terminal，observed reward固定为1，gamma=0.5。在U、D中都采用pi(L)=0.75、pi(R)=0.25。以下q数字都是固定的当前estimates。

| 实际next state | q_est(L) | q_est(R) | policy加权estimate | Expected Sarsa target |
|---|---:|---:|---:|---:|
| U | 6 | -2 | 4 | 1+0.5*4=3 |
| D | 2 | -2 | 1 | 1+0.5*1=1.5 |

在U，Sarsa会因next action不同而得到target 4或0；Expected Sarsa固定得到3。

在D，Sarsa会因next action不同而得到target 2或0；Expected Sarsa固定得到1.5。

但是环境仍可能把同一个当前(s,a)送到U或D，所以Expected Sarsa的target仍会在3和1.5之间变化。它没有把两个next states也平均成2.25。表中的50%环境概率是教学时给定的，标准Expected Sarsa的这次update没有使用这些概率。

即使next state固定，observed reward如果随机，Expected Sarsa target也可能随机。反过来，如果本例next state、reward、policy和q estimates全都固定，Expected Sarsa target就固定；估计更新过程中q或policy改变，target仍可能随训练改变。

## 记住expectation的范围

这里的expected是“给定已到达的next state，对next action求expectation”。它不是对环境的所有未来情况求完整expectation，也不代表能预知环境结果。

Expected Sarsa去掉的是next-action抽样带来的那部分随机性，不能据此宣称所有target noise或训练波动消失。

## 两项短检查，集中提交

1. 固定next state为U，保持上表q estimates、policy和gamma不变，只让本次observed reward在0和2之间随机变化。Expected Sarsa target可能是哪两个数？随机性来自哪里？
2. 若next-action policy改成100%选择L，在相同observed reward、next state和q estimates下，Sarsa与Expected Sarsa的target是否相同？即使如此，环境随机性是否仍可能存在？

无需代码或截图。正确解释后继续behavior policy、target policy与完整control例子。

## 来源

Sutton与Barto《Reinforcement Learning: An Introduction》6.6 Expected Sarsa：消除的是随机选择next action带来的variance。

https://incompleteideas.net/book/bookdraft2016sep.pdf

赵世钰教材第7章Box 7.4同样将next-action取样替换为policy加权expectation。U、D与具体数字为自编教学例子。

https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/3%20-%20Chapter%207%20Temporal-Difference%20Methods.pdf


## 后续反馈：action固定，environment仍可能随机

用户提交：“1、可能是3或5，来自rewards；2是吧，感觉有可能，但我不知道是啥”。

第1项随机性来源判断正确；target应为2或4。原policy加权estimate仍是4，gamma=0.5，bootstrap部分为2；新reward是0或2，替换原来的reward=1。因此target分别为0+2和2+2，不能将原reward继续留在target里。

第2项，在相同observed reward、next state和q estimates下，若policy确定100%选L，Sarsa与Expected Sarsa的target相同。因为Expected Sarsa的action加权平均只剩q_est(s',L)。用户尚未具体解释剩余环境随机性，继续针对该点补讲，不记为完整解释通过。

policy决定怎么选action；environment决定执行后收到什么reward、到达什么next state。确定policy并不保证environment给出的结果固定。以机器人每次都执行close、却可能抓稳或滑落为教学例子，可以将这种结果差异建模为environment随机性。

保留前文U、D的q estimates、gamma=0.5、reward=1，但next-action policy在两处都确定选L：实际到U时，Sarsa与Expected Sarsa的target都为1+0.5*6=4；实际到D时，二者都为1+0.5*2=2。同一次样本上两种target相同，重复采样时却仍可能因next state不同而变化。

若reward、next state、q estimates和policy也全部固定，target才在这组固定输入下固定；不要误解成Expected Sarsa一定会随机波动。

当前短检查（替代已批改的两项，不追加数字题）：机器人policy每次都执行close，结果有时holding、有时slipped。1. 这是否说明policy仍在随机选择action？2. Expected Sarsa会不会把holding和slipped两种环境结果都平均进这一次target？简短说明原因即可，无需代码或运行。
