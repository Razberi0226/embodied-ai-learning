# 2026-10-28学习包：第04课续学——exploration schedule与state-action coverage

今天继续第04课，先核对昨日最后的policy疑点，再进入两个新模块；不因日期变化自动标第04课完成，也不自动发布第05课。第03课RMSE统计维度与措辞复查并入本包末尾。昨日“好的”记为收到补讲，不视作新的独立作答。

## 全天清单与时间预算

| 环节 | 分钟 | 输入与成果 |
|---|---:|---|
| 闭卷回忆 | 5 | 先答本包第1、2项，核对policy概率、behavior return与on/off分类 |
| 两模块讲解 | 30 | A：epsilon schedule；B：state-action coverage与未知estimate |
| 休息 | 10 | 离开屏幕 |
| 核心与扩展实验 | 55 | 阅读Q-learning采样代码，先预测三种schedule，再只改warm start |
| 综合检查 | 15 | 集中提交第3—6项，含第03课RMSE间隔复习 |
| 学习记录 | 5 | 保留预测、设置、输出、理由与未解决问题 |

总计约120分钟，是计划量，不是已学习时间。用户可以按反馈调整。代码由导师提供并本地验证；用户主要读关键语句、预测、做少量修改和解释，不要求从零写。

## 起点与课程边界

已反馈通过：三种TD targets的计算、实际action与max target的区别、greedy evaluation指标选择。昨日待核对：P(good)=1来自改用确定policy的假设，不是连续选good自动得到；实际带exploration的behavior return不等于Q-learning的optimal target。

模块A、B属于当前L04的进一步应用，先完成开场短检查再按理解推进。文件发布不等于已学会。若开场仍有核心疑惑，先做针对性补讲，不强行进入后文。

## 复用同一个两步MDP

| state | action | reward | next state |
|---|---|---:|---|
| S | safe | 1 | T |
| S | shortcut | 0 | B |
| B | good | 2 | T |
| B | bad | -10 | T |

T真正terminal；gamma=0.9。environment确定，每episode最多两个transitions。greedy optimal路线是shortcut后good，discounted return为1.8；safe为1。这些是当前自编小MDP的解析性质。

今天固定Q-learning与其max target，主要改变behavior采样安排。这样不把算法差别与探索设置混在一起。

## 模块A：exploration schedule如何改变概率

在本例每个state有两个actions。唯一greedy action的概率为1-epsilon/2，另一个action概率为epsilon/2。q数值变化但greedy action未变、epsilon未变时，概率可以不变；greedy action翻转或epsilon改变时，概率可能改变。q并列时，本代码在并列greedy actions间均分概率。

epsilon schedule是我们指定的epsilon随episode编号变化的规则，不是用action出现次数直接替代概率。代码中的episode_index从0开始，每个episode内两个states使用同一个epsilon；它不是每个state自己的访问计数。

本包比较三种自编schedule：

1. fixed：epsilon始终为0.2。
2. linear：epsilon从0.2线性降到0.02，经过1500个episodes后维持0.02。
3. fast：episode_index=e时，epsilon=0.2/(e+1)。它很快变小，本实验用于检查有限预算内是否会错过经验，不保证所有环境下的好结果或收敛。

例如第100个episode对应e=99：fixed为0.2，linear仍在缓慢下降，fast为0.002。假设good始终唯一greedy，action概率直接由各自epsilon计算。

epsilon变小会减少直接exploration，但“以后更greedy”并不能保证此前已获得足够数据。尤其初始q全为0时，没有试到的action的estimate也可能为0。

### 例题：观测次数与概率不是同一个量

若某state中L仍是唯一greedy、epsilon=0.2，即使最近20次都抽中L，下一次P(L)仍为0.9。最近20次的样本频率为1，policy规定的概率仍为0.9。

## 模块B：state-action coverage决定看到了哪些经验

用N(s,a)记录某state-action pair被实际访问和update的次数。看q表时同时看N：q=0且N=0表示尚未采样，不能当作reward或真实value等于0。

off-policy允许behavior policy与target policy不同，仍需要采集对学习目标有用的经验。max target不会凭空知道未观察到的good reward。标准tabular收敛分析需要持续访问相关state-action pairs并满足learning-rate条件；本包只检查有限预算的结果，不把这些小实验当成收敛证明。

### 到达一个state后才能探索其中的action

本环境进入B必须先在S选shortcut，因此：

P(访问B中的bad)=pi(shortcut|S)*pi(bad|B)。

假设某一阶段S的唯一greedy action是safe，B的唯一greedy action是good，epsilon=0.2：两个所需action的概率都为0.1，这个pair每episode被访问的概率是0.01。

若之后S的greedy action变成shortcut而epsilon不变，进入B的概率变成0.9，B中的bad仍为0.1，访问(B,bad)的概率变成0.09。

这两个例子使用不同阶段的固定policy假设，不是声称整段training的分布不变。它说明同一个epsilon下，访问分布仍会因q ranking变化而变化。分支更长时，某些经验可能更难采集。

## 核心实验：fixed、linear、fast

输入：同目录“exploration_coverage.py”。Python标准库即可，默认每个seed训练2000 episodes，5个seeds，WARM_START=False，q初始化全0，alpha=1/N。

先读epsilon_at、choose_action、train中的max target、visits更新、evaluate_greedy。今天始终是Q-learning，变化的是behavior采样安排。evaluation冻结q并关闭exploration，不调用update。

运行前预测：哪种schedule更可能在某些seeds下缺少进入B或尝试某个action的经验？不要只根据最终epsilon大小排名。然后查看：

- N(S,shortcut)、N(B,good)、N(B,bad)。
- q(S,shortcut)与q(S,safe)，同时检查对应访问次数。
- 最终greedy路线和return，以及5个seeds中的个别失败。
- 实际environment steps。相同episode预算不保证相同transition预算；本包不据此声称哪种方法在固定transition预算下更快。

共享每个seed、episode、state对应的uniform numbers便于控制随机来源；q变化会改变actions，所以实际经验仍可能不同。

## 扩展实验：只改WARM_START

只把顶部WARM_START=False改成True。前4个episodes使用指定采样安排：safe；shortcut后good；shortcut后bad；shortcut后good。它们计入同一个2000 episodes总预算，不额外赠送经验，不手动填写真实q值。之后仍按原schedule运行，episode编号不重置。

这叫本包的forced exploration warm start。代码先实际执行、收到reward，再照常update。它不是加载一个训练好的q表。

先预测：对于很快缩小epsilon的fast schedule，初始几次有计划的采样可能怎样改变最终greedy路线？然后比较WARM_START开关前后，保留algorithm、MDP、初始化、gamma、alpha规则、seeds、总episodes和epsilon函数全部相同。

不能从这个小确定MDP推出四次经验就足以解决一般随机机器人任务。这里B到T的reward确定，一次该pair的观测就能给出其terminal reward；随机reward下还需要讨论更多样本。

## 运行方式与交付

本地直接用Python运行整份脚本；在线编译器可以复制整份代码，但不要求现在重跑Programiz。用户此前报告网页卡顿，导师不操作其Run按钮。可以先读代码与提交预测，再由导师展示标明来源的参考结果。

导师已检查实际交付脚本、两组WARM_START配置、固定采样的target传播、访问计数、terminal、budget与greedy evaluation不改q。参考结果保存在work目录，先预测再反馈，不记为用户已经运行。

## 六项检查，按阶段集中提交

### 开场先答第1、2项

1. 在B中good仍是唯一greedy action，epsilon固定0.2，连续20次抽中good。下一次P(good)会是多少？哪两个因素变化可能让这个概率改变？
2. 同一张q表，training的behavior在B以0.9/0.1选good/bad；greedy evaluation在B总选good。Q-learning把shortcut估到1.8时，实际指定shortcut、按training policy继续的expected return是否也一定是1.8？若两个算法最终greedy路线相同，判断on/off-policy应依据什么关系？可用昨天的0.72与1.8解释，不要求重新计算。

### 模块与实验后集中答第3—6项

3. 解释fast schedule为什么可能在有限预算内遗漏有用经验；从代码找出epsilon由episode编号决定的语句，并说明q=0、N=0能否证明该action真实value为0。给出运行前预测，再结合seed结果分析。
4. 假设S的唯一greedy action为safe、B的唯一greedy action为good，epsilon=0.2，访问(B,bad)的概率是多少？如果epsilon保持不变而S的greedy action改成shortcut，该概率怎样变？说明为什么这不违反“固定epsilon时同一greedy ranking给出固定action概率”。
5. WARM_START开启后，fast schedule是否更容易学到shortcut？比较5个seeds的coverage与greedy evaluation，说明只改了什么、是否额外增加训练预算，以及证据适用范围。用户未运行时可用导师提供的结果作解释，但须注明来源。
6. 第03课间隔复习：每次evaluation评估10个states，每个episode末evaluation一次，共100个episodes。每个RMSE平均多少个平方误差？整段有多少个RMSE数值？末段平均RMSE较低能否单独证明波动更小？

提交模板：第1—2项先用短段回答；之后第3—6项保留预测、必要输出和解释。无需从零代码或截图。用实际回答决定是否收尾L04、补讲或继续下一主题。

## 来源

- 赵世钰教材第7章：Sarsa、Q-learning、on/off-policy和采样条件。https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/3%20-%20Chapter%207%20Temporal-Difference%20Methods.pdf
- Sutton与Barto《Reinforcement Learning: An Introduction》：epsilon-greedy、exploration与tabular TD control。https://www.incompleteideas.net/book/bookdraft2018mar21.pdf

两步MDP、三种schedule与warm-start安排为本包自编教学配置，不是教材推荐的通用最优参数。


## 开场实际反馈与当前进入位置

用户第1项将good概率写成0.1，已直接纠正为good=0.9、bad=0.1；epsilon因素正确，behavior policy改变细化为当前epsilon-greedy实现中的q ranking或epsilon改变。此前已算对过，此次不为标签互换增加单题等待。

第2项解释实际behavior return会受到bad影响，正确；依据behavior/target policy关系判断on/off-policy，理由正确。进入两新模块，不再重复这项。

三个schedule在第1/100/1501个episode的epsilon分别为：fixed=0.2/0.2/0.2；linear=0.2/0.18812/0.02；fast=0.2/0.002/约0.00013324。episode编号从1叙述，代码e从0开始。

本轮先集中提交第3项核心预测及代码含义、第4项两阶段访问概率；接着看核心结果。第5项warm-start扩展与第6项RMSE间隔复查随后提交。此处记录教学反馈，不记为用户已运行实验。
