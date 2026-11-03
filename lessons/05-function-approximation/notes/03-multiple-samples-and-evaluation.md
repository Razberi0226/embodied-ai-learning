# 第05课：多个 samples 怎样共同影响 parameters？

实际发布日期：2026-10-30；接续2026-10-28已通过的预习。第01—04课已完成，本课发布不代表完成。不补发多天新课。

今天的核心问题：共享 parameters 时，怎样让一组 samples 整体拟合得更好，并判断这种改善能否扩展到未训练的位置？

## 学习安排与交付

| 环节 | 预计分钟 | 要完成的事 |
|---|---:|---|
| closed-book recall | 5 | 回忆 e=prediction-target；gradient=(e*x,e) |
| 模块A与B | 30 | 理解 batch gradient、整体 objective 与评估范围 |
| 休息 | 10 | 离开屏幕 |
| 核心与扩展实验 | 55 | 先预测，读关键代码，再改一个条件、比较结果 |
| 综合小测 | 15 | 5题集中提交 |
| 学习记录 | 5 | 写证据范围和剩余疑惑 |

120分钟是计划，不声称你实际已学习这么久。若卡住，先提交卡住的位置。

输入：上次的线性模型、单sample gradient与减alpha更新规则；今天的完整Python脚本。
成果：一张alpha对照表、一张coverage对照表、5题答案和一段结果解释。
运行：本地Python标准库即可，无GPU、安装包或远程机器需求。把同目录实验.py保存后在终端运行，也可先用导师参考输出读代码；不要求你从零写，不要求再次使用之前卡住的网页。导师未替你点击任何网页的Run。

完整脚本：[multiple_samples_experiment.py](../experiments/multiple_samples_experiment.py)
参考输出：[导师参考输出](../experiments/multiple_samples_reference_output.txt)。先写预测再打开参考输出，避免把看到的结果当预测。

## 模块A：从一个sample到一个batch

继续使用：
$\hat v(x,w)=w_1x+w_0,\qquad e_i=\hat v(x_i,w)-y_i.$

每个sample都有自己的error和gradient。今天用同一组旧parameters为所有samples计算gradient，再取平均：
$J(w)=\frac{1}{2N}\sum_{i=1}^{N}e_i^2,$
$\frac{\partial J}{\partial w_1}=\frac1N\sum_i e_i x_i,\qquad
\frac{\partial J}{\partial w_0}=\frac1N\sum_i e_i.$
$w^{new}=w-\alpha\nabla J(w).$

推导只多一步：求和的导数等于各项导数求和；1/N保留。不是把两个gradients相乘，也不是只更新error最大的sample。此处使用mean objective；若改成sum却保持alpha不变，更新幅度会随N变化，已改变实验条件。

这一步是batch gradient descent。若每看一个sample就立即改parameters，下一个sample会用已经改过的parameters，属于逐sample的更新过程；通常不等于本节的同一步平均gradient。相关算法区别可对照[Stanford CS229讲义第4—7页](https://cs229.stanford.edu/notes-spring2019/cs229-notes1.pdf)。我们的mean归一化与数字例子为教学设定。

### 例子：一个sample变好，另一个可能变差

两个samples为(0,1)、(1,3)，旧weights=(1,1)，alpha=.2。

| 情况 | weights | x=0的error | x=1的error | 两个samples的J |
|---|---|---:|---:|---:|
| 旧值 | (1,1) | 0 | -1 | .25 |
| 仅用x=1更新一次 | (1.2,1.2) | .2 | -.6 | .10 |
| 从旧值做一次batch更新 | (1.1,1.1) | .1 | -.8 | .1625 |

仅用x=1的gradient=(-1,-1)，所以两个parameters各增加.2。原来已经正确的x=0因为共享intercept而偏离target。batch的两个sample gradients是(0,0)和(-1,-1)，平均为(-.5,-.5)。

注意：batch不是保证每个sample每步都改善；它在当前点取整体J的gradient。合适的step可以降低整体J，而某个sample的error仍可能增大。也不能凭这一个例子宣布single-sample或batch普遍更好；这里使用的sample数量和每次更新处理的数据量不同。

### 阅读关键代码

~~~python
for x, target in data:
    error = predict(x, weights) - target
    grad_w1 += error * x
    grad_w0 += error
return grad_w1 / n, grad_w0 / n
~~~

weights在这个循环中没有被修改。循环结束后才执行一次step，这保证求的是同一个位置的平均gradient。

## 模块B：优化、sampling coverage与evaluation

今天构造一个已知reference：
$v_{ref}(x)=2x+1.$
x可以想成一维位置feature；不是任意state编号。训练targets由这个reference生成，方便核对误差。这是固定target的合成拟合实验，尚未运行RL environment，也没有学习action policy。

要区分三个问题：

| 问题 | 需要的证据 |
|---|---|
| parameters能否降低当前训练集的J？ | 同一训练集上的loss轨迹 |
| 未参与更新的位置是否预测更准？ | 在固定evaluation位置，用可靠reference计算误差 |
| 机器人是否更会完成任务？ | 还需要实际任务、policy和独立评测，本实验不提供 |

### learning rate改变step，不改变固定target

核心实验数据为x=(-1,0,1)，y=(-1,1,3)；旧weights=(0,0)，分别用alpha=.05、.2、2.2，都是40次batch updates。

每次update处理3个samples，因此每组使用120次sample，不是120条不同经验；也不是40个environment episodes。当前没有随机采样或噪声，不需要虚设seed。

先预测：.05与.2谁在40次内更接近reference？2.2是否仍然越大越快？说出理由，然后再看输出。适当alpha可能降低loss，过大的step可能越过合适区域并使loss放大；不能把上次确定性小例子中的快慢顺序推广到任意alpha。

### sampling coverage改变可获得的信息和有限预算下的学习

扩展实验只把训练位置从wide=(-1,0,1)改成narrow=(-.1,0,.1)，每组仍是3个samples、相同reference、alpha=.2、旧weights=(0,0)、40次updates。

两组共用evaluation positions=(-.5,.5,2)，它们都未参与训练。前两点在wide范围内，最后一点在wide范围外；三个点都在narrow范围外。这是刻意构造的coverage对照，不是所有现实测试分布的代表。

当|x|很小时，slope对应gradient e*x的幅度受feature缩小影响；同样40次updates，slope可能学得更慢。本例直线模型足以表示reference，有限预算下训练差异不能直接归因于模型容量不足。

不要用两个不同训练集各自的train loss直接排模型好坏，因为衡量的位置已不同。使用共同evaluation集合更容易比较，但有限几个evaluation点仍不能证明所有位置都准确。本实验也没有用evaluation结果反复挑配置后再假装它们是从未查看的最终test结果。

### 和RL targets接上，但先不跳到DQN

| target来源 | 一条sample的y是什么 |
|---|---|
| 本课合成实验 | 给定数值2*x+1 |
| episodic MC value prediction | 实际观察到的return G_t |
| TD value prediction | r+gamma*v_hat(s_next,w_old)，真正终止时为r |

representation、objective、update是不同问题：有一个function输出，不等于已经有了正确target。

在常见semi-gradient TD中，本次计算gradient时将已算出的TD target当常数，只对当前state的prediction求导；下一次更新target仍可随parameters变化。“这一步不对target求导”不等于“以后target永远不变”。这一点可核对[Harvard课程讲义2.2.2.2](https://hankyang.seas.harvard.edu/OptimalControlReinforcementLearning/value-rl.html)。当前课只建立接口，代码仍在拟合固定labels，不能宣称它已实现TD、DQN或机器人训练。

## 核心实验：预测、解释、少量修改

1. 先写alpha=.05、.2、2.2的结果预测。读batch_gradient与step：指出哪一行保证平均，哪一处保证两个parameters使用同一组旧weights。
2. 运行完整脚本，或根据导师参考输出逐行追踪。记录三组的update=0/5/20/40 train loss、最终weights和evaluation RMSE。若用参考输出，请注明“分析导师输出”，不写成亲自运行。
3. 仅把STEPS从40改为80，保留其他条件，再观察.05与.2；是否可以由有限结果声称“更大alpha永远更好”？若不方便运行，明确预测与尚未验证的部分，由导师协助检查。
4. 核对默认每组sample_uses应怎样计算；若guard提前停止，用实际updates计算，不虚报相同预算。程序的guard只是限制数值，不表示自动修好了不稳定更新。

建议约30分钟，重在解释和比较，不重复手抄代码。

## 扩展实验：只改coverage

1. 先预测wide/narrow哪个slope在40次updates后更接近2，哪个共同evaluation RMSE更低。
2. 还原STEPS=40；读C部分固定条件，运行或分析两组输出。记录weights、各自train loss、共同evaluation RMSE，指出哪些evaluation位置是interpolation、哪些是extrapolation。
3. 仅把NARROW_XS改成(-.2,0,.2)，保持其他条件，预测并检查slope学习变化；写明这仍未使全部evaluation位置落入训练区间。
4. 用2—3句话给出结论：当前变化由什么变量触发，证据只支持到什么范围；再用一个假想robot例子解释“只在狭窄位置采样，远处行为不能凭训练loss保证”。这是类比，不是本实验已测得robot结果。

建议约25分钟。

## 综合小测：5题集中反馈，先不给整套答案

1. 新例子：samples=(0,0)、(2,4)，旧weights=(1,0)，alpha=.1。求两个sample errors、平均gradient、一次batch后的weights。只计算这一次，不再重复上次的target1题。
2. 解释为何batch_gradient内不能边累加gradient边直接改weights；如果改了，是否仍是我们定义的同一步batch gradient？
3. 根据alpha实验，用一组具体loss或weights支持你的判断：更大alpha是否必定在固定预算下更好？这会不会改变给定training targets？
4. 根据coverage实验，narrow的train loss较小是否足以说明在x=2更准？应该比较什么量？再指出一个当前实验无法证明的更广泛结论。
5. 只做target接口迁移：当前state feature x=1，旧weights=(1,0)，next-state feature x_next=2，r=0，gamma=.9，未终止，alpha=.1。先从旧weights算TD target；把这一步target视为常数，求error和两个new parameters。如果这一步真正终止，target应改成什么？不用实现训练循环。

## 提交格式与验收

可以分两批：先交预测与第1—2题，实验和第3—5题稍后集中交；不用每个算术单独一轮。

~~~text
实际用时（未计时可写未计时）：
结果来源：本地运行 / 分析导师输出 / 二者都有
运行前预测：
alpha对照表：alpha、updates、sample_uses、最终weights、train loss、eval RMSE
coverage对照表：训练位置、weights、train loss、共同eval RMSE
只改一个条件的结果与解释：
小测1—5：
本课结论的适用范围：
仍不确定的地方：
~~~

完成以实际计算、关键代码解释、对照与证据范围为准；只说看懂或导师替你运行不算完整验收。当前仍停在L05，未答题不自动推进到DQN。

## 资料与课程对应

今日两组数值实验为自编教学设置。赵老师第8章的主线是从table到function representation，再连接objective、optimization与RL：[作者官方第8章](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/3%20-%20Chapter%208%20Value%20Function%20Methods.pdf)。单sample/batch gradient推导另用上文CS229讲义核对，semi-gradient接口用上文Harvard讲义核对；不声称这些数字实验原样来自赵老师视频。
