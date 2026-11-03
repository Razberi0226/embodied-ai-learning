# 第03课：MC target、value estimate与stochastic比较

日期：2026-10-27。数值实验由导师在本地运行，不计作用户亲自运行。

## 本次反馈

用户正确判断：terminal reward从固定1变为0或2各半后，E[R]=1保持不变，因此本trajectory上true state values不变。state value本来就是expected return。

用户将A的MC target算成2*0.2=0.4：这里误把learning rate乘进target，并跳过了从A到terminal reward的discount。0.4只对应假设old estimate为0，在D用target=2、alpha=0.2更新后的estimate。

本trajectory为A→B→C→D→T，observed rewards为[0,0,0,R]。因此R=2时：

- A的MC target = gamma**3 * 2 = 1.458。
- true v_pi(A) = gamma**3 * E[R] = 0.729。
- 如果old v(A)=0、alpha=0.2，new estimate = 0 - 0.2*(0-1.458) = 0.2916。

MC target不含learning rate；learning rate用于把old estimate更新为new estimate。使用教材current-target的error方向时，update前取减号，数值与加alpha*(target-current)等价。

用户担心fixed learning rate使estimate持续波动，机制判断正确，但TD同样采用fixed learning rate，因此这条理由本身不能证明MC的RMSE更高。MC sample target的expectation可以等于true state value，而单次target仍可能离它很远。

## 匹配预算的实验设置

discount rate=0.9；learning rate=0.2；initial estimates全0；每种方法300 episodes；episode-start frozen estimates；同一种子内共享reward序列。只比较n=1、2、3与MC的目标horizon。seeds为7、19、42、101、2026。

每个seed先计算最后50 episodes的平均RMSE，再跨5个seeds汇总。matched deterministic reference同样使用300 episodes，末50平均RMSE都在浮点精度附近。

| method | 5-seed mean score | standard deviation across seeds |
|---|---:|---:|
| n=1 | 0.186603 | 0.046438 |
| n=2 | 0.213513 | 0.051448 |
| n=3 | 0.225736 | 0.052562 |
| MC | 0.231107 | 0.052928 |

这5个seeds中，n=1的末50平均RMSE均最低。seed间standard deviation只描述这5次运行，不是普遍优劣证明；结果取决于reward设置、learning rate、预算、update scheme和metric。deterministic下最早达标与stochastic下较低的后期误差是不同问题。

## 当前仍需核对的三个点

1. 同一条R=2的trajectory，alpha从0.2改为0.1，A的MC target会改变吗？
2. 若old v(A)=1、alpha=0.1，本次MC target仍为1.458，new estimate是多少？
3. method X最后一次RMSE=0.01、末50平均RMSE=0.20；method Y最后一次RMSE=0.05、末50平均RMSE=0.10。若关心持续保持较低prediction error，选哪个？这两项排名为什么可以不同？

本节尚待上述理解检查，不能因导师跑完实验就记成用户掌握。无需从零写代码或立即操作网页。
