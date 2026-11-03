# 第04课：on/off-policy实验结果与反馈

日期：2026-10-27。以下为导师本地运行并核对过的参考结果，不记为用户运行提交。代码未改变，本轮读取已经验证的结果，没有重新运行Programiz。

## 本轮作答反馈

- 第1项此前已通过：good/bad概率0.9/0.1，Sarsa所学policy包含exploration。
- 第2项：带exploration的shortcut expected discounted return=0.72，正确；Sarsa偏向safe、Q-learning偏向shortcut，预测正确。之后在B总选good时概率变为1/0，对应return=0+0.9*2=1.8，用户给出的仍为0.72需纠正。
- 第3项：max参与target构造、environment_step真正执行action，解释正确；关闭exploration后的使用效果看greedy evaluation return，正确。
- 第4项：epsilon下降后Sarsa对shortcut评价上升、greedy选择可能改变，方向正确。更准确的机制是bad被选择的概率下降，good的reward本身始终是2。

## 两种后续policy，两个不同的action values

在B按good/bad概率0.9/0.1行动：q_pi(S,shortcut)=0.9*(0.9*2+0.1*(-10))=0.72。之后总选good：q_*(S,shortcut)=0.9*2=1.8。safe的value为1。

Sarsa学习与behavior policy对应的action values；Q-learning的max target面向optimal continuation。Q-learning估shortcut为1.8，并不表示保留epsilon=0.2的behavior policy实际每次都在B选good。

epsilon=0.05时，P(good)=0.975、P(bad)=0.025；Sarsa所针对的shortcut expected discounted return变为0.9*(0.975*2+0.025*(-10))=1.53，大于safe的1。因此其greedy选择转向shortcut。

## 实验配置

每种方法每个seed训练10000 episodes；gamma=0.9；q全零初始化；alpha=1/N，N按state-action pair计数；training指标取末1000个episodes的discounted return均值；greedy evaluation冻结q并关闭exploration。seeds为7、19、42、101、2026。

两个epsilon配置为0.2和0.05，扩展只改变epsilon。同seed两方法共享每个episode/state的uniform numbers；实际trajectories不保证相同。每episode至多两步，实际environment steps另行报告，不宣称固定transition预算下的效率比较。

## 跨5个seeds的描述性汇总

| epsilon | 方法 | training末窗口均值 | greedy evaluation | 最终greedy路线（5/5 seeds） | shortcut estimate均值 | 平均environment steps |
|---|---|---:|---:|---|---:|---:|
| 0.2 | Sarsa | 0.989520 | 1.000000 | safe | 0.723625 | 11017.0 |
| 0.2 | Q-learning | 0.654960 | 1.800000 | shortcut -> good | 1.799719 | 18975.2 |
| 0.05 | Sarsa | 1.449200 | 1.800000 | shortcut -> good | 1.531082 | 19491.4 |
| 0.05 | Q-learning | 1.449200 | 1.800000 | shortcut -> good | 1.799738 | 19656.4 |

training均值是完整episodes的样本平均，包含S处和B处的exploration，不等于指定shortcut后的单个action value，也不要求恰好等于理论expectation。有限窗口会有采样波动。

epsilon=0.2时，本次设置下Sarsa的training均值较高，Q-learning导出的greedy policy在关闭exploration后return较高。应按使用目标选择评价方式，不能直接宣布某方法在所有任务上更好。

epsilon=0.05时两者末窗口training均值恰好相同：晚期两者选择相同的greedy actions，并共享action抽样numbers，因此该窗口经历相同trajectories。它们的shortcut estimates仍不同，分别接近1.53与1.8；相同行为不表示学习目标相同。

## 每个seed的原始统计

| epsilon | seed | 方法 | training末窗口均值 | greedy evaluation | greedy路线 | q(S,shortcut) | steps |
|---|---:|---|---:|---:|---|---:|---:|
| 0.2 | 7 | Sarsa | 0.952800 | 1.000000 | safe | 0.695769 | 11040 |
| 0.2 | 7 | Q-learning | 0.527600 | 1.800000 | shortcut -> good | 1.799799 | 18973 |
| 0.2 | 19 | Sarsa | 0.974800 | 1.000000 | safe | 0.758559 | 10999 |
| 0.2 | 19 | Q-learning | 0.668400 | 1.800000 | shortcut -> good | 1.799597 | 18928 |
| 0.2 | 42 | Sarsa | 1.011200 | 1.000000 | safe | 0.822180 | 11064 |
| 0.2 | 42 | Q-learning | 0.658800 | 1.800000 | shortcut -> good | 1.799800 | 18997 |
| 0.2 | 101 | Sarsa | 0.996000 | 1.000000 | safe | 0.562617 | 10963 |
| 0.2 | 101 | Q-learning | 0.649600 | 1.800000 | shortcut -> good | 1.799800 | 18997 |
| 0.2 | 2026 | Sarsa | 1.012800 | 1.000000 | safe | 0.778999 | 11019 |
| 0.2 | 2026 | Q-learning | 0.770400 | 1.800000 | shortcut -> good | 1.799599 | 18981 |
| 0.05 | 7 | Sarsa | 1.477600 | 1.800000 | shortcut -> good | 1.536157 | 19633 |
| 0.05 | 7 | Q-learning | 1.477600 | 1.800000 | shortcut -> good | 1.799813 | 19633 |
| 0.05 | 19 | Sarsa | 1.453600 | 1.800000 | shortcut -> good | 1.557030 | 19616 |
| 0.05 | 19 | Q-learning | 1.453600 | 1.800000 | shortcut -> good | 1.799626 | 19616 |
| 0.05 | 42 | Sarsa | 1.365600 | 1.800000 | shortcut -> good | 1.528770 | 19729 |
| 0.05 | 42 | Q-learning | 1.365600 | 1.800000 | shortcut -> good | 1.799815 | 19729 |
| 0.05 | 101 | Sarsa | 1.426800 | 1.800000 | shortcut -> good | 1.516937 | 19742 |
| 0.05 | 101 | Q-learning | 1.426800 | 1.800000 | shortcut -> good | 1.799815 | 19742 |
| 0.05 | 2026 | Sarsa | 1.522400 | 1.800000 | shortcut -> good | 1.516516 | 18737 |
| 0.05 | 2026 | Q-learning | 1.522400 | 1.800000 | shortcut -> good | 1.799624 | 19562 |

## 最后两项概念检查

1. Q-learning已把shortcut估到约1.8，但实际behavior policy在B仍以0.9概率选good、0.1概率选bad。从S指定shortcut后，实际expected discounted return应按0.72还是1.8理解？说明policy的区别。
2. epsilon=0.05时两算法最终greedy路线相同，是否足以判断它们属于同一种on/off-policy类型？分类应看最终路线，还是behavior policy与target policy的关系？

无需额外算数或运行。回答后据理解收尾当前模块；L03 RMSE统计维度等间隔复查保留。


## 补讲：action概率什么时候改变？

2026-10-27，用户问action概率是否实时改变，以及为何多次选good就变成1。

我们这个epsilon-greedy policy由当前q estimates的greedy ranking和epsilon决定。可以每次选action时重新计算，但概率不一定每次都改变。固定epsilon=0.2、good始终是唯一greedy action时，P(good)=0.9、P(bad)=0.1；连续多次抽到good，下一次仍按0.9/0.1抽样。q数值更新如果不改变greedy action，也不改变这里的概率。

概率可能随学习变化：初始化q全相同时，本代码对两个greedy ties均分概率；估计更新后若good成为唯一greedy则分布变为0.9/0.1；若greedy action翻转分布会翻转；若人为将epsilon调整到0.05，概率变为0.975/0.025。

此前“之后总选good”指另设一个确定policy，P(good)=1、P(bad)=0；greedy evaluation在good为唯一greedy action且epsilon=0时也有这个分布。这是对比后续policy的假设或evaluation设置，不是通过多次选择good自动把概率变成1。

相应地，0.72属于原探索policy，1.8属于之后总选good的policy。暂不新增题，原最后两项解释等待用户准备好后继续。
