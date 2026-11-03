# 第04课续学：on-policy、off-policy与两步control实验

日期：2026-10-27。用户上课返回后明确要求“继续学习”。本次从已通过的三种TD targets与Expected Sarsa随机性范围继续，不重复基础定义。这是当前学习链的续学模块，不将已发布当作已完成，也不宣称已经学习满120分钟。

## 上次的结束位置

用户正确指出抓取结果的差异来自environment，并正确判断Expected Sarsa不会把holding、slipped两个next states都平均进这一次target。“不在模型里”已当场修正为“不在这次expectation的范围内”：这些states仍可以属于MDP，Expected Sarsa使用实际到达的state，对其next actions求平均。

该处判断通过；措辞精度保留间隔复习。之前L03的RMSE统计维度复查保留，不借新模块的发布将其自动标为完成。

## 本次清单与交付

- 模块A：behavior policy与target policy；理解Sarsa和Q-learning的差异。
- 模块B：区分带exploration的training表现与冻结q后的greedy evaluation。
- 核心实验：同一个两步MDP、相同初始化和episode预算，比较Sarsa与Q-learning。
- 扩展实验：只将EPSILON从0.2改为0.05，其他设置不变。
- 本轮集中提交4项预测或解释；先预测，再看本地参考结果或自行运行。代码由导师提供，不要求从零编写。

建议本轮用30—45分钟读关键代码、预测和分析；这只是估计，按实际反馈调整。今天此前已进行了多个模块，不重新给本续学模块贴完整120分钟标签。

输入是本讲义和同目录“on_off_policy_two_step.py”；预期成果是解释policy角色、读懂target与实际action的区别，并判断两种evaluation设置回答的不同问题。

## 模块A：两种policy角色

behavior policy（pi_b）决定实际怎样选择actions、采集经验。

target policy（pi_T）是学习其value或希望改进、得到的policy。“target policy”与“TD target”不是同一个词：前者是policy，后者是一次update里的数值目标。

- on-policy：学习针对的policy与生成经验的policy相同。
- off-policy：允许学习针对的policy与生成经验的policy不同。这个名称不要求每一次实际action都不同，也不要求使用旧数据或两个神经网络。

| 方法 | behavior policy | 学习所针对的policy / values |
|---|---|---|
| 标准Sarsa control | 例如当前epsilon-greedy policy | 同一个带exploration的policy的action values，并不断改进policy |
| Q-learning | 可以是epsilon-greedy policy | 通过Bellman optimality target学习optimal action values，target使用greedy规则 |

两者都可以在training时explore。“有exploration”本身不能判断on/off-policy。

Sarsa target读取实际next action的estimate；Q-learning target读取max estimate。Q-learning的max用于构造target，不强迫behavior policy下一步实际执行argmax action。

这些是tabular情形下的学习目标；有限预算的estimates不保证已经等于真实values或optimal values。Q-learning仍需要足够覆盖state-action pairs，不能理解为随便收集一点数据就一定学好。

## 模块B：一个两步MDP

初始state为S，terminal为T。所有environment transitions和rewards都确定：

| 当前state | action | reward | next state |
|---|---|---:|---|
| S | safe | 1 | T |
| S | shortcut | 0 | B |
| B | good | 2 | T |
| B | bad | -10 | T |

每个episode最多两个transitions。gamma=0.9。这个MDP是自编教学例子，借鉴cliff-walking中exploration会影响behavior表现的思想，不是教材原始grid world或真实机器人实验。

本实验刻意使用确定environment，便于单独观察policy exploration的作用。前面讲过环境也可能随机，并不意味着所有环境必须随机。

training使用epsilon-greedy：以epsilon概率在全部actions中均匀explore；否则选greedy。某state有两个actions、greedy action唯一时，epsilon=0.2对应greedy概率0.9，另一个action概率0.1。

假设已经学到B中good比bad好：Sarsa在学习S的shortcut价值时，会受到后续policy偶尔选择bad的影响；Q-learning使用B中最大的estimate构造target。因此两个方法在S的greedy选择可能不同。请先自己计算和预测，不提前把实验排序当成答案。

## training与evaluation分别回答什么？

代码都使用discounted return，避免把gamma是否计入混在比较里。

1. training指标：保留epsilon=0.2，取末1000个training episodes的discounted return平均。它描述学习末期仍在explore时的实际表现；这期间estimates仍在更新。
2. greedy evaluation：training结束后冻结q、关闭exploration，使用greedy actions跑完episode。它描述当前greedy policy的执行效果，不进行update。

因为environment确定、evaluation采用固定tie规则，同一张冻结q表的greedy rollout结果确定，无需重复许多相同rollouts来制造样本量。

两个指标可以对算法给出不同排序。判断哪种policy更合适，需要先明确使用时是否保留exploration。不能根据training return更高就直接宣布部署时的greedy policy也更好。

## 运行与代码阅读

代码只依赖Python标准库，默认10000 episodes、5 seeds，每个episode最多两步，不进行网页操作。此前用户报告Programiz卡顿；本轮可以先读代码和预测，由导师提供参考输出，不要求用户立刻重新运行网页。

若在本地运行，在代码目录执行Python脚本即可；在线编译器如以后自愿使用，可整份复制，不需要安装库。

重点阅读：

1. `a_b = choose_action(...)`：先选定实际next action。
2. `next_value = q[next_state][a_b]` 与 `max(q[next_state])`：两种target的唯一核心差别。
3. `environment_step(next_state, a_b)`：实际执行的是原先选定的a_b，max没有改变它。
4. `evaluate_greedy`：冻结q，只执行动作，不调用update。

这里使用在线transition update，不沿用L03为比较n-step targets而采用的episode开头冻结整张value表的教学设定。

learning rate为每个state-action pair自己的alpha=1/N，其中N是该pair累计update次数。不要把结果解释成固定alpha实验；本轮只改epsilon。

## 控制变量与证据范围

两种方法使用相同MDP、gamma、全零初始化、alpha规则、10000 episodes和5个seeds。每个seed中共享按episode和state排列的uniform random numbers；因为estimates和actions可能不同，实际trajectories、returns仍可能不同，不宣称两者采集了完全相同的经验。

代码报告实际environment steps。一个方法可能更常走两步shortcut，因此相同episode预算不保证相同transition预算。本实验比较固定episode预算下的结果，不据此声称谁在固定environment-step预算下学习更快。

每个seed的结果分别保留，再做描述性汇总。仅限这个小MDP和指定设置，不能推出Sarsa或Q-learning在所有任务上更好。

## 扩展实验

只改顶部`EPSILON = 0.05`，保留全部其他配置。先预测探索失误变少后，Sarsa对shortcut的评价是否改变，以及它在S的greedy选择是否可能改变。再对照两个方法的training与greedy evaluation。

导师会核对代码与两组配置，但在用户预测前不提前给出扩展结果。导师运行不记为用户已独立运行或解释通过。

## 四项集中提交

1. 当B的greedy action已经是good时，epsilon=0.2下，good与bad各以多大概率被选中？Sarsa所针对的后续policy和Q-learning target中的greedy规则是否相同？
2. 用这些概率计算：带exploration时，从S选择shortcut后的expected discounted return是多少？如果之后在B总选good，又是多少？据此预测Sarsa和Q-learning各自更可能把safe还是shortcut作为S的greedy action。
3. 从脚本中指出：哪一行构造Q-learning的target，哪一行真正执行next action？如果实际选中bad，max是否会把它偷偷改成good？再说明training指标与greedy evaluation哪个更适合回答“关闭exploration后当前policy能拿多少return”。
4. 把epsilon改成0.05，预测Sarsa对shortcut的评价如何变化、S的greedy选择是否可能改变。这个预测只改变一个变量；不要求从零编码或提交截图。

可以用四段短答案集中提交。用户尚未运行时先讨论预测，再看标明来源的导师结果。

## 来源

- 赵世钰教材第7.4.2节：behavior policy、target policy、Sarsa的on-policy与Q-learning的off-policy解释。
  https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/3%20-%20Chapter%207%20Temporal-Difference%20Methods.pdf
- Sutton与Barto，第6章Example 6.6 Cliff Walking：exploration条件下的online表现与greedy policy的区别。
  https://www.incompleteideas.net/book/bookdraft2018mar21.pdf

本节两步MDP、实验代码和数字为自编材料。


## 按需回顾：exploration与exploitation

2026-10-27，用户在提交实验预测前请求复习exploration。此处是复习，不算新课完成，也不额外增加题目。

exploitation按当前q estimates选择最大的action，以利用当前已有信息；exploration让其他actions也有机会被尝试，获取经验、修正可能不准确的estimates。当前估计较低不保证真实value较低，未充分尝试的action可能被低估。

标准epsilon-greedy以1-epsilon概率选择greedy action，以epsilon概率在全部actions中均匀选择。exploration分支也可能抽中greedy action，因此epsilon不等于非greedy action最终被选中的概率。

两个actions L/R，q_est(L)=5、q_est(R)=2，epsilon=.2：80%概率直接greedy选L；20%均匀抽样又分别贡献10%给L和R。最终P(L)=.9，P(R)=.1。已知B中good为唯一greedy action时，同样得到P(good)=.9、P(bad)=.1。

exploration是policy选择action的机制，和environment在执行后的随机reward/transition不同。Sarsa和Q-learning都能使用epsilon-greedy；on/off-policy取决于behavior与target policy的关系，而不是是否explore。原四项预测保留，等待用户准备好后继续。


## 按需回顾：expected discounted return

用户已正确给出第1项good/bad概率0.9/0.1，并指出Sarsa所学policy包含exploration。第2项请求术语解释，尚未提交计算。

一条实际trajectory的discounted return为G=r1+gamma*r2+gamma**2*r3+...；expected discounted return再对可能的G按概率求平均。先给每条路径中的未来rewards乘折扣，再对不同路径加权。

本例从S指定shortcut：第一步reward0到B；随后good路径的G=0+0.9*2=1.8，bad路径的G=0+0.9*(-10)=-9。在good/bad概率0.9/0.1的policy下，expected discounted return的待算表达式为0.9*1.8+0.1*(-9)。

从S指定shortcut、之后遵循policy pi时，这个expected discounted return正是q_pi(S,shortcut)。一次episode提供一个实际G，取expectation得到对应action value。

原第2项继续：算出加权平均；若之后总选good，判断相应return；与safe的1比较，再预测两个方法在S的greedy选择。其余原题保留，不新增任务。
