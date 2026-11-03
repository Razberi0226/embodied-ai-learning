# 第04课续学：MC的边界，以及三种TD target

发布日期：2026-10-27。承接已通过的action-value加权、Sarsa单次update与terminal例子。这是按用户要求继续的关联小节，不代表完整L04已完成。中文解释，技术术语采用英文；不要求从零编码。

## 1. MC能否看成TD的特例？

在有限episode的n-step return框架中，可以把MC看成一直用到真正terminal的那个端点。MC与TD(0)依然有区别：TD(0)通常用一步reward加后续estimate进行bootstrapping；MC用实际观察到的完整return，不进行bootstrapping。

设episode在时刻T真正terminal，当前时刻为t。未到terminal时，n-step state-value target为：

$$G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}+\gamma^n v_{\mathrm{est}}(s_{t+n}),\qquad t+n<T.$$

当n足以覆盖剩余episode，即t+n>=T时：

$$G_t^{(n)}=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k+1}=G_t.$$

terminal之后没有bootstrap项，target就是MC return。不需要真的把n取为无穷大，覆盖剩余episode即可。这里首先说的是target的一致；完整算法是否相同还取决于visit规则、learning rate和update安排。

例如从当前state出发，还会收到rewards [0, 1, 2]并真正terminal，gamma=0.9。任意n>=3对应的target都是0+0.9*1+0.9**2*2=2.52。n=1或2时通常还需要末端estimate。

因此可采用这句准确表述：one-step TD和MC是n-step return框架的两个端点。这个联系本身不能推出两者谁总是更快或更准确。本文只讨论真正terminal，时间预算截断需要另外判断bootstrap。

## 2. 新问题：next action怎样进入TD target？

下面固定一组当前q estimates，比较三种规则怎样处理同一个observed reward和next state。比较公式，不是三个算法的训练性能实验。

记next state为s'，实际next action为a'；pi是本小节中采集经验时遵循的policy。暂假设离散actions，可以列出每个action的概率与estimate。

| 方法 | 非terminal的TD target | 处理next action的方式 |
|---|---|---|
| Sarsa | r + gamma*q_est(s', a') | 使用policy实际选中的a' |
| Expected Sarsa | r + gamma*sum_a pi(a\|s')*q_est(s', a) | 对该policy的next actions做加权平均 |
| Q-learning | r + gamma*max_a q_est(s', a) | 取当前最大的next-action estimate |

三者真正terminal时的target都只有r。统一沿用赵老师教材的符号方向：TD error=current estimate-target；new estimate=current estimate-alpha*TD error。

## 3. 一个完整例子

next state有L、R两个actions，q_est(L)=4，q_est(R)=1，pi(L)=0.9，pi(R)=0.1。这与epsilon=0.2、从所有actions均匀exploration的epsilon-greedy相符。已经实际选中R。observed reward=0，gamma=0.9。

- Sarsa：target=0+0.9*1=0.9。
- Expected Sarsa：先算policy平均0.9*4+0.1*1=3.7，因此target=0+0.9*3.7=3.33。
- Q-learning：target=0+0.9*max(4,1)=3.6。

Expected Sarsa的3.7是对action values加权，3.33才是target。它没有让机器人在下一步同时执行L和R，也不要求实际执行每个action后再平均。

若current estimate=2、alpha=0.2，三种规则各自算出的new estimates依次为1.78、2.266、2.32。它们都从同一个旧值2独立计算；不能把第一个结果接着当第二个方法的旧值。

## 4. 为什么要对next actions求expectation？

固定这组q estimates、reward和next state，让a'重复按同一个pi抽样。Sarsa有90%的机会得到target 3.6，10%的机会得到target 0.9。它的平均target为0.9*3.6+0.1*0.9=3.33，等于Expected Sarsa在该next state算出的target。

Expected Sarsa消掉的是这一步next-action抽样带来的随机性。reward和next state若是随机的，target仍然可能随机；有限次Sarsa样本平均也不保证恰好等于3.33。

## 5. behavior policy与target policy的联系

- behavior policy决定实际怎样选择actions、采集哪些经验。
- target policy是学习所针对的policy。
- 标准Sarsa用正在遵循的policy选出的next action进行update，是on-policy。
- 本节Expected Sarsa对同一个policy求expectation，也按on-policy理解；Expected Sarsa还可以设计off-policy版本，本节暂不展开。
- Q-learning可以一边按epsilon-greedy探索，一边用max构造target，学习朝向optimal action values；它的target使用greedy规则，不要求实际next action也greedy，这是典型的off-policy机制。

Q-learning对next actions取max，并没有对环境的next states取max；observed reward和next state仍来自实际经验。最大的estimate不保证是真实最优值。固定alpha、epsilon的短实验也不自动保证收敛到q_*。

## 6. 可运行演示

同目录“three_td_targets.py”只用Python标准库，打印固定输入下的targets和单次updates，以及两个控制变量的变化。代码不训练机器人、不运行长episode，也不操作在线编译器。

阅读顺序：先找compare_targets中的三个分支，再看update_from_target，最后看三组演示。用户无需从零编写，暂时无需在Programiz重新运行。导师本地验证结果单独保存至work目录。

两个控制变量演示：

1. 保持q estimates、policy、reward和gamma不变，只把实际next action从R改为L。
2. 保持q estimates、reward、gamma和实际next action R不变，只把epsilon从0.2改为0.6，按标准epsilon-greedy重新计算policy。

## 7. 三题集中提交

1. 新例子：next state非terminal；q_est(L)=6、q_est(R)=-2；pi(L)=0.75、pi(R)=0.25；实际next action为R；observed reward=1、gamma=0.5。分别计算Sarsa、Expected Sarsa、Q-learning的TD targets。无需再逐个计算new estimates。
2. 保持第1题的policy、q estimates、reward和gamma不变，只把实际next action改为L。哪种方法的target会改变？Expected Sarsa的加权平均是否要求下一步实际执行L、R两种actions？
3. 某次Q-learning update使用max对应L，但采集经验的policy实际选择R，是否允许？简述behavior policy与target构造的区别。再判断：Expected Sarsa是否消除了环境reward和state transition的随机性？

提交格式：第1题三个targets；第2题改变的方法及一句解释；第3题两个判断及理由。不要求写代码或提交运行截图。

## 教材来源

- 赵世钰《Mathematical Foundations of Reinforcement Learning》第7章：7.2 Sarsa、Box 7.4 Expected Sarsa、7.3 n-step Sarsa、7.4 Q-learning。作者上传全文：https://www.researchgate.net/publication/385891312_Mathematical_Foundations_of_Reinforcement_Learning
- 官方第7章文件：https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/3%20-%20Chapter%207%20Temporal-Difference%20Methods.pdf
- Sutton与Barto《Reinforcement Learning: An Introduction》第6、7章：https://www.incompleteideas.net/book/bookdraft2018mar21.pdf

数字和练习为本节自编教学例子，不是论文或教材实验结论。
