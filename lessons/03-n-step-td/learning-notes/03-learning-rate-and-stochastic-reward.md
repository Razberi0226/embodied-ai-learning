# 第03课learning rate对照与stochastic reward续学

日期：2026-10-27。结果由导师在本地执行已交付代码得到，不计作用户亲自运行。

## 用户的运行前预测

- ALPHA从0.2降为0.1，A首次为正的episode不变，认为信息传播机制没变。
- 首次达到RMSE<0.02更晚，认为update幅度更小。

两项在本实验中均正确。应精确区分“首次出现正estimate”与“达到指定accuracy”：前者在本例由transition距离、n和frozen update scheme决定；learning rate仍会影响estimate的变化幅度和达标速度。结论限于当前positive learning rate、zero initialization、deterministic reward及其他固定条件。

## 控制条件与结果

固定trajectory A→B→C→D→T；observed rewards为[0,0,0,1]；discount rate=0.9；initial estimates全0；每episode使用开始时的frozen estimates；预算80 episodes。只改变learning rate。

| 方法 | A首次为正（两种alpha相同） | RMSE<0.02首次episode，alpha=0.2 | RMSE<0.02首次episode，alpha=0.1 |
|---|---:|---:|---:|
| n=1 | 4 | 37 | 75 |
| n=2 | 2 | 24 | 50 |
| n=3 | 2 | 22 | 46 |
| MC | 1 | 17 | 36 |

每次本地执行设置5秒超时。未点击网页运行按钮。

## 下一阶段：stochastic reward

恢复alpha=0.2，保持discount rate=0.9、zero initialization、frozen update scheme。此阶段deterministic reference与stochastic设置都使用300 episodes；避免用上一阶段的80-episode末值直接推断reward randomness的影响。

只有terminal reward改变：R=0或2，各一半概率。每个seed先生成一份共享reward序列；n=1、2、3与MC各自读取同一份。用5个seeds，记录末50 episodes平均RMSE及跨seed描述性统计。参考代码的stochastic模式已增加matched deterministic reference。

由于E[R]=1，true state values仍为[0.729,0.81,0.9,1]。以D为例，某次sample return可以是0或2，但true state value是期望1。固定policy、相同trajectory仍可有random returns。

MC target直接包含这次实际terminal reward；使用bootstrapping的target还取决于后续value estimates。这带来不同的noise响应和滞后，不能仅从deterministic收敛速度推断stochastic误差排序。

## 待提交的三项预测与解释

1. R改为0或2各半后，true v(A)会不会改变？给出依据。
2. 本次R=2时，A的MC target是多少？这个sample是否说明true v(A)变了？
3. 预测MC在stochastic设置中仍会得到最低的末50 episodes平均RMSE吗？先说明直觉与理由，后续依据结果修正；不要求猜对。

无需从零写代码，也无需立即操作网页。参考结果已经由导师检查，但仍等待用户解释，不把教材/代码发布或导师运行当作已掌握。
