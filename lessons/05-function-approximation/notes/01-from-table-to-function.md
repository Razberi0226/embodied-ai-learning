# 第05课预习：从table到function

日期：2026-10-28。用户在第03、04课核心目标收尾后明确要求“继续学一点明天的”。本节是约15—20分钟的预习，实际用时未报告；不是提前发布完整120分钟学习包，也不把第05课标为完成。

## 今天只理解一个变化

tabular method为每个state或state-action pair分别保存estimate。function approximation把estimate表示为由state/features和一组参数共同决定的函数。

为只讨论表示方法，先看给定policy的state value。后续相同思想可以用于action value q_hat(s,a,w)；当前不用神经网络、不更新policy、不做完整TD训练。

## 为什么需要新的表示？

小MDP可以给每个state存一格。机器人state若包含连续关节位置、速度等，可能的states极多，逐个存表会受存储与采样覆盖限制。function approximation希望用参数化函数在多个states之间共享结构，对未直接采样的输入也给出estimate。

能够输出estimate，不保证输出准确；generalization需要合适的features、模型结构和数据支持。

## 一个最小线性例子

假设只使用state中的一个位置feature x，作为教学简化。x具有数值含义；它不是任意分配的state ID。

用线性函数表示：

$$\hat v(s,w)=w_1 x+w_0.$$

参数w=(w_1,w_0)。例如w=(2,1)，就是v_hat=2*x+1。features可写为phi(s)=(x,1)，于是v_hat=phi(s)^T w。这里w有两个参数，不需要为每个x另存一个独立estimate。

此处的输出仍是value estimate，不是action概率。改变value表示与改变epsilon-greedy规则是两件事。

## table更新与共享参数的影响

起初在x=0、1、2三个states的estimates为1、3、5。现在希望x=1处estimate由3变为4。

- tabular示例：只把x=1的那一格改成4。
- function示例：人为指定将intercept w_0从1改成2，slope w_1仍为2，函数变为2*x+2。

| x | 修改前 | 只修改table中x=1 | 改共享参数后的function |
|---|---:|---:|---:|
| 0 | 1 | 1 | 2 |
| 1 | 3 | 4 | 4 |
| 2 | 5 | 5 | 6 |

同一组w服务于多个states，因此一次参数改变可能影响多个estimates。本例所有输入都受intercept改变影响；不宣称任意参数更新都一定改变所有states。

这次w_0的改变是人为选定，用于观察表示的差别，不是已经推导或执行某种TD/gradient update。正式课再学习怎样根据sample target和loss更新w。

## generalization与影响其他states

旧函数在未单独存过的x=1.5处也能输出2*1.5+1=4。它是在利用所选的函数结构给出estimate，不能据此断言真实value就是4或这个state已经被采样过。

共享参数可以把经验带到其他states，也可能让某些states的误差变大。如果未知其他states的真实values，只看到预测跟着改变，就无法判断它们是否更准确。

features的选取也很关键。若x只是任意state编号，相邻编号不自动表示物理相似、value相近；用位置feature表达数值关系，是本例的建模假设。

## 可运行演示

同目录“shared_parameters_demo.py”只打印上述table和function对比，使用Python标准库，没有训练循环。用户不需要从零写代码，也不要求立刻运行Programiz。

代码里的new_w=(2,2)是明确给定的对照参数，不是假装模型已经学到了正确值。导师在本地核对演示输出，但不记作用户已运行。

## 三项预习检查，集中提交

1. 对另一个未列在table中的位置x=0.5，旧函数2*x+1给出多少？将intercept从1改到2后，新函数给出多少？为什么不修改table里的这一格，function的输出也能改变？
2. 如果只观察到x=1处的estimate更接近其target，能否据此证明x=0、x=2的estimates也更准确？说明还缺什么信息。
3. 在x=1.5处能算出estimate=4，是否说明模型真的见过这个state，或者真实value一定为4？简述estimate与证据的区别。

本节通过后只记录预习目标，不自动把完整第05课标为完成。用户可以随时停止预习，明天从实际反馈继续。

## 教材对应

赵世钰《Mathematical Foundations of Reinforcement Learning》第8.1节Value representation: From table to function，式(8.1)。本节数字与参数对照为自编例子。

https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/3%20-%20Chapter%208%20Value%20Function%20Methods.pdf


## 预习实际作答反馈（2026-10-28）

用户第1项给出旧1、新2，并解释参数共享；共享参数机制理解正确，数值应为旧2*0.5+1=2、新2*0.5+2=3。计算要包含2x这一项，已直接纠正，不增加重复算术题。

第2项“不能，缺少另外两个和target的差距”方向正确：一个state更接近其target不证明其他states更准确，还需其他states的误差证据。target本身是否可靠也要依具体任务判断。

第3项否认estimate=4即可证明见过state或真实value=4，判断正确。“只是经验判断”进一步精确为：按所选features、模型结构和parameters计算的estimate；是否有实际样本经验要看数据，准确性要看评估。本例参数改变是人为指定的对照，尚未执行参数学习。

本次预习核心目标通过，整节L05尚未完成。暂时无待交作业；下一正式课从sample target、loss与线性模型参数更新继续，不重复整段预习。不声称用户亲自运行或从零编写代码。
