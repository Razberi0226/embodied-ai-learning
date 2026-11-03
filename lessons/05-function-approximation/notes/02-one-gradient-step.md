# 第05课预习续：一次fixed-target gradient step

日期：2026-10-28。用户明确要求“再推进一点点”。本节约10—15分钟，实际时长未报告。只学习一个sample、一个固定target下的loss与parameter update，不是完整第05课，也不提前展开完整DQN。

## 从人为改参数到根据target更新

继续使用state-value estimate：

$$\hat v(x,w)=w_1x+w_0.$$

给定sample位置x=1、当前参数(w_1,w_0)=(2,1)，所以prediction=3。现在给定training target y=5，希望该sample的prediction靠近5。

这里y只是给定的训练target，不假设它一定等于真实v_pi。后续会把它接到observed MC return或TD target上；本节先固定y，求导时它是常数。

## squared loss

定义prediction error：

$$e=\hat v-y.$$

方向与此前current estimate减target的约定一致。当前只是fixed-target回归，因此不把e直接叫作TD error。

选单sample loss：

$$L(w)=\frac12(\hat v-y)^2=\frac12e^2.$$

1/2使求导后的表达式更简洁。本例e=3-5=-2，loss=2。loss衡量prediction与当前target的距离，不是reward，也不是action概率。

## gradient的两个分量

gradient由loss对各个parameters的偏导数组成：

$$\frac{\partial L}{\partial w_1}=e x,\qquad \frac{\partial L}{\partial w_0}=e.$$

原因是loss对prediction的导数为e，而prediction对w_1的导数为x、对w_0的导数为1。将这两层影响相乘，就得到上式。

因此在同一个error下，不同parameters可能得到不同的gradient；它还取决于对应feature。当前x=1，两个gradient都为-2。

## gradient descent的一步

用learning rate alpha=0.1，两个分量都在同一组旧参数上计算，然后更新：

$$w_1^{new}=w_1-\alpha e x,$$
$$w_0^{new}=w_0-\alpha e.$$

代入本例：

```text
prediction = 2*1+1 = 3
error      = 3-5 = -2
gradient   = (-2*1, -2) = (-2, -2)
new w1     = 2-0.1*(-2) = 2.2
new w0     = 1-0.1*(-2) = 1.2
new prediction = 2.2*1+1.2 = 3.4
new loss       = 0.5*(3.4-5)**2 = 1.28
```

这次在给定alpha下，该sample loss由2降到1.28。不是直接把prediction改成target；是先改parameters，再重新计算prediction。两个parameters各增加0.2，所以x=1处prediction总共增加0.4。

这只说明本例该sample更接近给定target，不证明其他states更准或目标是真实value；任意更大的alpha也不保证loss下降。正式学习包再用控制变量实验检查这些问题。

## 代码与更新设定

同目录“fixed_target_update.py”提供完整可运行函数，没有长训练循环，不要求从零写或运行网页。

```python
prediction = w1 * x + w0
error = prediction - target
new_w1 = w1 - alpha * error * x
new_w0 = w0 - alpha * error
```

这里prediction和error先从旧参数计算一次，两项gradient都来自同一位置。这是本节的同时gradient step；先改一个parameter再重新求error属于另一个顺序更新过程，不混称为同一步公式。

用户此前已理解参数共享，因此本次只聚焦loss到parameter update。输出仍是value estimate，alpha控制参数步长，不涉及epsilon或实际action selection。

## 两项关联检查，集中提交

1. 换一个sample：x=2、target y=4、旧(w_1,w_0)=(1,0)、alpha=0.1。计算旧prediction、error、new w_1、new w_0和new prediction；解释为什么本次w_1的改变量是w_0的两倍。
2. 按本节这套update公式，如果sample的x=0，这次sample能否直接改变w_1？为什么？w_0是否仍可能改变？不用写代码。

两项反馈后再决定预习收尾或是否按用户要求继续。完整L05仍需多sample学习、泛化检查和与TD targets的衔接，不能把一次例题通过当作整课完成。

## 对照资料

- 赵世钰教材第8.1节的线性value representation、第8.2节的objective与optimization主线。本节是先固定target的回归基础，数字与半平方loss定义为自编教学设定。
  https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/3%20-%20Chapter%208%20Value%20Function%20Methods.pdf
- Stanford CS229 linear regression讲义，LMS与single-example squared-loss gradient推导，约第4页。
  https://cs229.stanford.edu/notes_archive/cs229-notes-all/cs229-notes1.pdf


## 实际作答反馈：error符号与feature乘积

用户旧prediction=2正确；按本节e=prediction-target，target4时e=2-4=-2。若改用target-prediction和加号可定义等价更新，但当前统一沿用减alpha乘current-target，避免混用。

两个gradient为grad_w1=e*x=(-2)*2=-4，grad_w0=e=-2。新w1=1-.1*(-4)=1.4，新w0=0-.1*(-2)=.2，新prediction=1.4*2+.2=3。w1改变量.4，是w0改变量.2的两倍；共同error乘入两项，位置feature x额外乘入slope梯度。

用户关于x=0时w1不能直接更新、w0仍可能因非零error更新的解释正确，不重复追问这一项。

当前只做一个反方向短检查：保持x=2、旧weights(1,0)、alpha=.1，target改为1。旧prediction仍为2。集中提交两组数值：1.e及两个gradients；2.两个new parameters。无需重算prediction或编写代码，不新增概念。


## 反方向短检查反馈（2026-10-28，已收尾）

用户说明之前没看到target从4改为1，并自行给出：e=1、grad_w1=2、grad_w0=1、new w1=0.8、new w0=-0.1，全部正确。前一条仍答target4的数值按数据未注意到处理，不另认定概念误区。

prediction偏高时，正error通过减alpha乘gradient降低当前sample的prediction；本次两项parameter来自同一组给定旧weights。零feature判断与本次更新计算均已通过，这段预习收尾，不新增待交题。完整L05尚未完成，下次从多sample拟合和受控实验继续；未确认用户运行代码，实际用时未知。
