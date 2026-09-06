# `backward_propagation` 为什么能用于重建：数学命题、证明与实现边界

本文讨论 [huo/art.py](https://github.com/zhiqwang/huo/blob/main/huo/art.py) 中的 `forward_propagation`、`backward_propagation` 和 `art`。问题是：反向过程没有直接实现严格 ART 的逐射线更新公式，为什么仍然可能重建图像？这种现象能否得到数学证明？

可以证明的是：沿射线回填残差具有明确的投影纠错作用；在特定离散模型中，它与 ART 完全等价；更一般的正反投影算子只要满足适当的稳定性条件，也可以迭代求解。当前代码是否对某组参数满足这些条件，则需要另行验证。下面的条件性定理不构成当前实现对任意输入都能精确重建的证明。

全文以实数运算描述算法，暂不计浮点舍入误差。“精确恢复”指恢复离散模型中的真解；把真实物体离散化和把物理测量近似为线积分所产生的误差，属于另一层问题。

**定义 1：代码对应的离散算子。**

固定几何参数和角度 $\theta$，设图像边长为 $P$，探测器数量为 $d$，把图像展平为 $x\in\mathbb R^n$，其中 $n=P^2$。定义

$$
A_\theta:\mathbb R^n\to\mathbb R^d,
\qquad
B_\theta:\mathbb R^d\to\mathbb R^n,
$$

分别为正向投影和 `backward_propagation`。在采样坐标固定时，插值和求和都是线性的，因此可以用矩阵表示这两个算子，无须在实现中显式存储矩阵。

| 符号 | 含义 | 代码对应 |
| --- | --- | --- |
| $x$ | 当前图像，展平后的向量 | `img` |
| $b_\theta$ | 某角度的测量投影 | `sinogram[:, :, :, i]` |
| $A_\theta x$ | 预测投影 | `forward_propagation(...)` |
| $r_\theta=b_\theta-A_\theta x$ | 未归一化的投影残差 | 除以 `img_len` 之前的 `res` |
| $B_\theta r_\theta$ | 残差回填到图像空间 | `backward_propagation(...)` |
| $L>0$ | 统一的归一化长度 | `param.img_len` |
| $P_+$ | 逐分量映射 $z_j\mapsto\max(z_j,0)$ | `torch.clamp(img, min=0)` |

因此，`art` 每处理一个角度所做的操作是

$$
\boxed{
x^{k+1}
=P_+\left[x^k+\alpha B_{\theta_k}
\left(b_{\theta_k}-A_{\theta_k}x^k\right)\right],
\qquad \alpha=\frac1L.
}
\tag{1}
$$

式 (1) 是本文与实际实现的连接点。后续讨论其他步长时，会明确把 $\alpha$ 当作一般正数。

这里的 $B_\theta$ 接收残差。不能仅凭函数名把它解释为 $A_\theta^{-1}$，也不能把“正向投影后再反投影”直接解释为恢复原图。

**命题 1：反向坐标公式识别的是像素所属的射线。**

记 $S=\mathrm{SOD}>0$，$D=\mathrm{SDD}>0$。在未旋转的扫描器坐标系中，源点为 $(0,-S)$，坐标为 $u$ 的探测器位置为 $(u,D-S)$。

对于世界坐标中的点 $p=(X,Y)$，先旋转回扫描器坐标系：

$$
\begin{pmatrix}X'\\Y'\end{pmatrix}
=R_{-\theta}\begin{pmatrix}X\\Y\end{pmatrix}.
$$

假设讨论区域位于源点前方，即 $S+Y'>0$。定义探测器坐标映射

$$
u_\theta(p)=\frac{D X'}{S+Y'}.
\tag{2}
$$

那么，同一条从源点射向探测器 $u$ 的射线上的所有点，都满足 $u_\theta(p)=u$。

**证明。** 用到源点的距离 $s>0$ 参数化射线。在扫描器坐标系中，射线上的点为

$$
X'(s)=\frac{s u}{\sqrt{u^2+D^2}},
\qquad
Y'(s)=-S+\frac{sD}{\sqrt{u^2+D^2}}.
$$

代入式 (2)：

$$
u_\theta(p(s))
=\frac{D\,s u/\sqrt{u^2+D^2}}
{sD/\sqrt{u^2+D^2}}
=u.
$$

因此式 (2) 在射线上保持常数。证毕。

在 `backward_propagation` 中，旋转矩阵执行 $R_{-\theta}$。如果归一化图像坐标为 $\xi=X'/\texttt{img\_end}$、$\eta=Y'/\texttt{img\_end}$，那么

$$
\frac{D\xi}{\eta+S/\texttt{img\_end}}
=\frac{DX'}{Y'+S}.
$$

随后除以 `detr_end`，就得到了探测器上的归一化采样坐标。最后的插值是在读取该探测器位置的残差。这里的几何公式是正确的；像素坐标是否与正向离散网格一致，是后文单独讨论的问题。

为了分析这一几何操作，下面先引入与离散矩阵区分开的连续算子 $\mathcal A_\theta$、$\mathcal B_\theta$。

**定义 2：连续线积分与沿射线回填。**

固定角度 $\theta$ 和有限探测器坐标区间 $U$。设 $\Omega$ 是位于源点前方的有界图像区域；只讨论满足 $u_\theta(p)\in U$ 的像素位置。记 $\Gamma_{\theta,u}$ 为正向积分实际使用的、位于 $\Omega$ 内的射线段，允许它被积分窗口截断，其长度为

$$
\ell_\theta(u)=\int_{\Gamma_{\theta,u}}1\,ds.
$$

取使下述积分有定义的有界可测函数 $f$、$g$，定义

$$
(\mathcal A_\theta f)(u)
=\int_{\Gamma_{\theta,u}} f(p)\,ds,
\qquad
(\mathcal B_\theta g)(p)=g(u_\theta(p)).
\tag{3}
$$

后者表示沿射线均匀回填。它是当前反向实现所近似的几何操作，但尚未包含图像网格与探测器网格之间的离散插值。

**命题 2：连续正反投影复合是射线长度乘法。**

在定义 2 的条件下，

$$
\boxed{
(\mathcal A_\theta\mathcal B_\theta g)(u)
=\ell_\theta(u)g(u).
}
\tag{4}
$$

**证明。** 根据命题 1，积分路径上恒有 $u_\theta(p)=u$，所以

$$
\begin{aligned}
(\mathcal A_\theta\mathcal B_\theta g)(u)
&=\int_{\Gamma_{\theta,u}}g(u_\theta(p))\,ds\\
&=g(u)\int_{\Gamma_{\theta,u}}1\,ds\\
&=\ell_\theta(u)g(u).
\end{aligned}
$$

证毕。

令 $\mathcal D_\theta$ 表示乘以 $\ell_\theta(u)$ 的算子。若只考虑长度严格为正的射线，且有关函数除以长度后仍在算子定义域内，则

$$
\mathcal C_\theta
=\mathcal B_\theta\mathcal D_\theta^{-1}
\quad\Longrightarrow\quad
\mathcal A_\theta\mathcal C_\theta=I.
\tag{5}
$$

例如，$0<\ell_{\min}\leq\ell_\theta(u)$ 可以保证有界数据除以长度后仍然有界。式 (5) 说明 $\mathcal C_\theta$ 是一个右逆：先回填归一化投影、再沿射线积分，可以重现该投影。

不过，反过来的 $\mathcal C_\theta\mathcal A_\theta$ 一般不是恒等算子。单角度存在沿各射线积分为零的非零函数 $h$；对这样的 $h$，

$$
\mathcal C_\theta\mathcal A_\theta h=0\ne h.
$$

所以“重现投影”和“恢复图像”是两个不同的数学命题。

**命题 3：连续模型中，归一化回填可以收缩当前角度的残差。**

令 $r=b_\theta-\mathcal A_\theta f$，暂时不做非负截断。采用代码形式的更新

$$
f^+=f+\frac1L\mathcal B_\theta r.
$$

则更新后的残差满足

$$
\boxed{
r^+(u)
=\left(1-\frac{\ell_\theta(u)}L\right)r(u).
}
\tag{6}
$$

**证明。** 利用线性性和式 (4)：

$$
\begin{aligned}
r^+
&=b_\theta-\mathcal A_\theta
\left(f+\frac1L\mathcal B_\theta r\right)\\
&=r-\frac1L\mathcal D_\theta r.
\end{aligned}
$$

逐点展开即得式 (6)。证毕。

若 $\ell_\theta(u)=L$，该射线的残差一次被消除。若 $0<\ell_\theta(u)/L<2$，该射线的残差幅值减小。

要得到统一的范数收缩率，需要更强的条件。假设几乎处处有

$$
0<\ell_{\min}\leq\ell_\theta(u)
\leq\ell_{\max}<2L,
$$

并且 $r\in L^2(U,du)$。定义

$$
q=\max\left\{
\left|1-\frac{\ell_{\min}}L\right|,
\left|1-\frac{\ell_{\max}}L\right|
\right\}<1.
$$

由式 (6) 两边平方并积分，得到

$$
\|r^+\|_{L^2(U)}\leq q\|r\|_{L^2(U)}.
\tag{7}
$$

仅有每条射线上的严格减小，不足以声称存在统一的 $q<1$；例如，有效长度趋于零时，收缩因子可以趋于 $1$。长度为零的射线无法通过图像更新纠正其非零测量值。

如果使用每条射线自身的长度，即 $f^+=f+\mathcal C_\theta(b_\theta-\mathcal A_\theta f)$，则更新一次就满足该角度的全部投影约束。不过，这一步仍然可能改变其他角度的投影。单角度结论不能代替多角度收敛证明。

**命题 4：等权重离散模型中，残差除以长度就是 ART。**

先推导严格的单射线更新。设 $a_i\in\mathbb R^{1\times n}$ 为一个非零行向量，测量约束为 $a_i x=b_i$。考虑最小改动问题

$$
\min_\delta\frac12\|\delta\|_2^2
\quad\text{满足}\quad
a_i\delta=b_i-a_i x=:r_i.
$$

其拉格朗日函数为

$$
\mathcal L(\delta,\nu)
=\frac12\|\delta\|_2^2-\nu(a_i\delta-r_i).
$$

一阶条件给出 $\delta=\nu a_i^T$；代入约束得到

$$
\boxed{
\delta=\frac{r_i}{\|a_i\|_2^2}a_i^T.
}
\tag{8}
$$

目标函数严格凸，所以这是唯一的最小改动。它就是到超平面 $a_i x=b_i$ 的欧氏正交投影修正。ART/Kaczmarz 的标准背景可参见 [Kak 与 Slaney，第 7 章，§7.1–7.2](https://engineering.purdue.edu/~malcolm/pct/CTI_Ch07.pdf)。

现在假设一条射线恰好经过像素集合 $J_i$，其中有 $N_i$ 个像素，且每个像素贡献相同的长度权重 $h>0$：

$$
a_{ij}=
\begin{cases}
h,&j\in J_i,\\
0,&j\notin J_i.
\end{cases}
$$

于是 $\|a_i\|_2^2=N_i h^2$，射线的总长度为 $L_i=N_i h$。式 (8) 在射线内化为

$$
\boxed{
\delta_j
=\frac{r_i h}{N_i h^2}
=\frac{r_i}{L_i},
\qquad j\in J_i.
}
\tag{9}
$$

这证明：在等权重模型中，均匀回填 $r_i/L_i$ 与严格 ART 完全一致。分母里的平方和与分子中的权重发生了约分。

若统一改用 $L$ 而非 $L_i$，则同一模型下得到的是松弛 ART：

$$
x^+=x+\lambda_i\frac{r_i}{\|a_i\|_2^2}a_i^T,
\qquad \lambda_i=\frac{L_i}{L}.
\tag{10}
$$

对满足该射线约束的任意真解 $x_*$，令 $e=x-x_*$。由于 $r_i=-a_i e$，直接展开平方可得

$$
\boxed{
\|e^+\|_2^2
=\|e\|_2^2
-\lambda_i(2-\lambda_i)
\frac{|a_i e|^2}{\|a_i\|_2^2}.
}
\tag{11}
$$

因此 $0<\lambda_i<2$ 时，到真解的距离不增加；该射线有非零残差时严格减小。这也是式 (6) 中长度比例条件的离散对应。

这些结论依赖于等权重假设以及逐射线修正。若同角度的各射线支撑互不重叠，它们不会改动彼此涉及的像素，可以同时处理而仍与依次处理等价。存在支撑重叠和插值时，这种等价性一般不成立。

**命题 5：当前离散实现应当用各自的插值权重建模。**

令 $z_{\theta,i,t}$ 表示第 $i$ 条射线上的第 $t$ 个正向采样点；令 $\varphi_j(z)$ 表示 `grid_sample` 对第 $j$ 个图像像素的双线性插值权重，包含零填充约定。正向算子的矩阵元素是

$$
(A_\theta)_{ij}
=\Delta s\sum_t\varphi_j(z_{\theta,i,t}),
\qquad
\Delta s=\frac{L}{P\,\texttt{lat\_sampling}}.
\tag{12}
$$

另一方面，令 $\widehat p_j$ 为反向函数实际赋给第 $j$ 个像素的物理位置，$\psi_i(u)$ 为探测器一维线性插值权重，则

$$
(B_\theta)_{ji}
=\psi_i(u_\theta(\widehat p_j)).
\tag{13}
$$

**证明。** 正向每个采样值都可以写为 $\sum_j\varphi_j(z)x_j$，乘以 $\Delta s$ 并沿射线求和，交换有限求和次序即得式 (12)。反向插值直接是 $\sum_i\psi_i(u)r_i$，因此得到式 (13)。证毕。

式 (12) 与式 (13) 使用不同的权重构造。当前实现没有构造或保证 $B_\theta=A_\theta^T$，也没有实现式 (8) 所要求的逐行平方范数归一化。这里不能仅凭两个操作都叫“投影”就认定它们互为转置。

离散算子同样存在精确的残差恒等式：若不做截断，

$$
r_\theta^+
=\left(I-\alpha A_\theta B_\theta\right)r_\theta.
\tag{14}
$$

连续式 (4) 在这里通常不再是对角恒等式，因为图像插值和探测器插值会使不同射线的离散信息混合。因此，式 (6) 对几何原理的证明，不能当成式 (14) 对实际矩阵的逐射线收缩证明。

要讨论离散重建，必须转而分析图像误差的传播矩阵。下面给出这一分析所需的定理。

**定理 1：固定线性迭代的精确恢复条件。**

设 $A\in\mathbb R^{m\times n}$、$B\in\mathbb R^{n\times m}$ 固定，$\alpha>0$，并存在 $x_*$ 使 $b=Ax_*$。考虑不带截断的迭代

$$
x^{k+1}=x^k+\alpha B(b-Ax^k).
\tag{15}
$$

对任意初值 $x^0$ 都有 $x^k\to x_*$，当且仅当

$$
\boxed{\rho(I-\alpha BA)<1.}
\tag{16}
$$

这里 $\rho$ 表示谱半径，即所有特征值模的最大值。这个定理讨论固定的 $A,B$，例如把所有角度一起使用的迭代；它不是对当前逐角度循环的直接替换。

**证明。** 令 $e^k=x^k-x_*$，则

$$
e^{k+1}=(I-\alpha BA)e^k=:Te^k,
\qquad e^k=T^k e^0.
\tag{17}
$$

若 $\rho(T)<1$，在复数域上将 $T$ 化为 Jordan 标准形。对于特征值 $\lambda$、幂零部分 $N$ 的 Jordan 块，块的幂是有限个形如 $\binom{k}{j}\lambda^{k-j}N^j$ 的项之和。$|\lambda|<1$ 时，每项趋于零；$\lambda=0$ 的块则在有限次幂后为零。因此 $T^k\to0$，于是任意 $e^0$ 都有 $e^k\to0$。

反过来，若对所有实数初始误差都有 $T^k e^0\to0$，取标准基向量可知 $T^k$ 的每一列都趋于零，因此 $T^k\to0$。若 $T$ 有 $|\lambda|\geq1$ 的复特征值及对应非零特征向量 $v$，则 $T^k v=\lambda^k v$ 不趋于零，矛盾。所以 $\rho(T)<1$。证毕。

由 $\lambda\in\sigma(BA)$ 可知 $1-\alpha\lambda\in\sigma(T)$，且

$$
|1-\alpha\lambda|^2
=1-2\alpha\operatorname{Re}\lambda+\alpha^2|\lambda|^2.
$$

因此式 (16) 等价于对 $BA$ 的每一个特征值都有

$$
\boxed{
\operatorname{Re}\lambda>0,
\qquad
0<\alpha<\frac{2\operatorname{Re}\lambda}{|\lambda|^2}.
}
\tag{18}
$$

此处必须包含全部特征值；零特征值会破坏“任意初值恢复同一真解”的结论。秩亏情况下，若只要求在指定不变子空间内收敛，需要另外限定初值和解的选择，不能直接忽略零特征值。

这给出了“不必是转置也能重建”的形式化回答：式 (16) 的推导从未使用 $B=A^T$，只使用了数据一致性和误差传播矩阵的稳定性。不匹配正反投影的固定迭代理论及其更一般的子空间情形，可参见 [Dong 等，§2：The BA Iteration](https://arxiv.org/html/1902.04282#S2)。

例如，$A=I$、$B=2I$、$\alpha=1/4$ 时，$T=I/2$，误差每步减半。$B$ 在此既不是 $A^T$ 也不是 $A^{-1}$，但仍然精确收敛。

还应区分渐近收敛与每一步都变好。即使 $\rho(T)<1$，非正规矩阵的 $\|T\|_2$ 也可能大于 $1$，某些误差会先增长再衰减。谱半径条件本身不保证逐步单调减小欧氏误差。

若测量包含噪声而不再满足 $b=Ax_*$，式 (16) 仍可保证收敛到唯一固定点，因为 $BA$ 此时可逆：

$$
x_\infty=(BA)^{-1}Bb.
\tag{19}
$$

它满足 $B(Ax_\infty-b)=0$。这一般不同于普通最小二乘的正规方程 $A^T(Ax-b)=0$，也不意味着恢复了无噪声真值。式 (19) 只适用于本定理的固定算子迭代。

**定理 2：逐角度、固定顺序重复扫描的收敛条件。**

设角度为 $\theta_1,\ldots,\theta_s$，每个角度的算子固定，且存在共同真解 $x_*$ 满足

$$
b_{\theta_i}=A_{\theta_i}x_*,\qquad i=1,\ldots,s.
$$

暂时去掉 `clamp`。每个角度的图像误差满足

$$
e^+=T_\theta e,
\qquad T_\theta=I-\alpha B_\theta A_\theta.
\tag{20}
$$

若始终按 $\theta_1,\ldots,\theta_s$ 的顺序重复扫描，定义

$$
M=T_{\theta_s}\cdots T_{\theta_1}.
\tag{21}
$$

则对任意初值恢复同一 $x_*$ 的充要条件为

$$
\boxed{\rho(M)<1.}
\tag{22}
$$

**证明。** 令 $e^{[t]}$ 表示第 $t$ 轮开始时的误差。逐次代入式 (20) 得到

$$
e^{[t+1]}=Me^{[t]},
\qquad e^{[t]}=M^t e^{[0]}.
$$

对 $M$ 使用定理 1 中的矩阵幂论证即可得到轮次边界上的充要条件。一轮中的中间误差则是有限个固定前缀乘积 $T_{\theta_j}\cdots T_{\theta_1}$ 作用于 $e^{[t]}$。这些前缀的范数有有限上界，因此轮次边界误差趋于零时，中间误差也趋于零。证毕。

注意乘积的次序不能任意调换；不同角度的更新矩阵一般不交换。也不能把当前逐角度算法的稳定性直接替换为全角度求和更新的稳定性。

若存在非零向量 $z$ 使所有 $A_\theta z=0$，则每个 $T_\theta z=z$，从而 $Mz=z$。因此有公共零空间时，式 (22) 在整个图像空间上不可能成立。这个限制反映的是信息不足，并非反投影实现独有的问题。

**推论：每轮改变角度顺序时，需要控制变化矩阵的乘积。**

设第 $t$ 轮采用排列 $\pi_t$，对应矩阵为 $M_{\pi_t}$。则

$$
e^{[t]}
=M_{\pi_{t-1}}\cdots M_{\pi_0}e^{[0]}.
\tag{23}
$$

一个足够条件是存在共同的矩阵诱导范数和常数 $q<1$，使所有允许的排列均满足 $\|M_\pi\|\leq q$。由次乘性，式 (23) 立即给出 $\|e^{[t]}\|\leq q^t\|e^{[0]}\|$。

仅证明每个排列各自的谱半径小于 $1$，一般不足以证明切换后的稳定性。例如取

$$
U=\begin{pmatrix}1/2&2\\0&1/2\end{pmatrix},
\qquad
V=\begin{pmatrix}1/2&0\\2&1/2\end{pmatrix}.
$$

两者的谱半径都是 $1/2$，但

$$
VU=\begin{pmatrix}1/4&1\\1&17/4\end{pmatrix},
\qquad
\rho(VU)=\frac94+\sqrt5>1.
$$

这个矩阵反例说明切换分析的必要性；它并不是在声称仓库的某两个角度恰好产生了 $U,V$。

**引理：非负截断不会增加候选值到非负真解的距离。**

如果 $x_*\geq0$，则对任意 $z\in\mathbb R^n$，

$$
\|P_+(z)-x_*\|_2\leq\|z-x_*\|_2.
\tag{24}
$$

**证明。** 对每个分量，若 $z_j\geq0$，两边该分量相同；若 $z_j<0$，则 $|0-x_{*,j}|=x_{*,j}\leq x_{*,j}-z_j=|z_j-x_{*,j}|$。平方后求和即得结论。证毕。

式 (24) 比较的是截断前后的候选值，并没有把它们与上一轮图像比较。因此它不能单独保证迭代误差递减，也不能让定理 2 自动适用于带 `clamp` 的代码。

**定理 3：带非负截断的逐角度迭代，一个明确的充分条件。**

仍假设存在共同非负真解 $x_*$。定义 $T_\theta$ 如式 (20)，并令

$$
\mathscr D=
\left\{\operatorname{diag}(d_1,\ldots,d_n):0\leq d_j\leq1\right\}.
$$

如果存在 $q<1$，使每个允许的角度排列 $\pi$ 和任意 $D_1,\ldots,D_s\in\mathscr D$ 都满足

$$
\boxed{
\left\|
D_sT_{\theta_{\pi(s)}}\cdots D_1T_{\theta_{\pi(1)}}
\right\|_2\leq q,
}
\tag{25}
$$

那么式 (1) 在重复扫描时对任意初值都收敛到 $x_*$，并且每轮边界有 $\|e^{[t]}\|_2\leq q^t\|e^{[0]}\|_2$。

**证明。** 标量函数 $h(z)=\max(z,0)$ 单调且为 $1$-Lipschitz。因此任意实数 $a,b$ 都可以写成

$$
h(a)-h(b)=d(a-b),\qquad 0\leq d\leq1.
$$

当 $a\ne b$ 时，取差商即可；当 $a=b$ 时任取 $d\in[0,1]$。

对某一步，截断前的候选值为

$$
z=x+\alpha B_\theta(b_\theta-A_\theta x)
=x_*+T_\theta(x-x_*).
$$

逐分量使用差商，结合 $P_+(x_*)=x_*$，存在 $D_k\in\mathscr D$ 使

$$
e^{k+1}=P_+(z)-P_+(x_*)=D_kT_{\theta_k}e^k.
\tag{26}
$$

沿一轮扫描相乘，再使用式 (25)，得到 $\|e^{[t+1]}\|_2\leq q\|e^{[t]}\|_2$。归纳即得轮次边界上的收敛。由于 $\|D_k\|_2\leq1$，且一轮的角度数有限，中间步骤同样趋于真解。证毕。

式 (25) 是保守的充分条件，不是必要条件；它也比不带截断的谱半径条件强。本文没有验证当前实现满足式 (25)。它的作用是明确指出：要把线性证明扩展到当前非线性更新，需要控制截断与各角度更新复合后的行为。

**反例：非负正反投影权重不足以保证恢复真解。**

取

$$
A=I_2,
\qquad
B=\begin{pmatrix}1&2\\2&1\end{pmatrix},
\qquad
x_*=b=\begin{pmatrix}1\\2\end{pmatrix}.
\tag{27}
$$

$A$、$B$ 都具有非负元素，数据完全一致，且 $A$ 的解唯一。但 $BA=B$ 有特征值 $3$ 和 $-1$。因此，对任意 $\alpha>0$，$I-\alpha BA$ 都有特征值 $1+\alpha>1$，不带截断的迭代不可能对所有初值稳定收敛。

加入非负截断也不能保证正确。令

$$
\bar x=\begin{pmatrix}5\\0\end{pmatrix}.
$$

直接计算：

$$
b-A\bar x=\begin{pmatrix}-4\\2\end{pmatrix},
\qquad
B(b-A\bar x)=\begin{pmatrix}0\\-6\end{pmatrix}.
$$

于是对任意 $\alpha>0$，

$$
P_+\left[\bar x+\alpha B(b-A\bar x)\right]
=P_+\begin{pmatrix}5\\-6\alpha\end{pmatrix}
=\bar x\ne x_*.
\tag{28}
$$

这是一个错误的非负固定点。它否定了“只要把残差用非负权重回填，再截断为非负，就必然重建正确”的一般性命题。这个例子是代数反例，不是针对当前 CT 几何参数的数值实验。

这些定理应用到仓库时，还需要核对以下具体事实。

**实现事实 1：反向图像坐标存在比例偏差。**

[huo/radon.py](https://github.com/zhiqwang/huo/blob/main/huo/radon.py) 和坐标准备代码使用

$$
h=\frac LP,
\qquad
\texttt{img\_end}=\frac{L-h}{2}.
$$

正向 `grid_sample(..., align_corners=True)` 下，第 $j$ 列像素中心的归一化坐标为

$$
\xi_j^{\mathrm{true}}=-1+\frac{2j}{P-1},
\qquad j=0,\ldots,P-1,
$$

其物理坐标是 $\texttt{img\_end}\,\xi_j^{\mathrm{true}}$。

反向代码调用 `affine_grid` 时未指定 `align_corners`。现代 PyTorch 的默认值为 `False`，对应未旋转的归一化坐标为

$$
\xi_j^{\mathrm{false}}
=-1+\frac{2j+1}{P}
=\frac{P-1}{P}\xi_j^{\mathrm{true}}.
$$

默认行为以及角点和像素中心的约定见 [PyTorch：affine_grid](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.affine_grid.html)。

两个坐标轴都具有同一比例，旋转与统一缩放可交换。因此在当前的 `img_end` 换算下，式 (13) 中的反向物理位置为

$$
\widehat p_j=\frac{P-1}{P}p_j.
\tag{29}
$$

这会改变像素对应的探测器位置。把网格约定统一，是消除该项几何偏差的方式；它仍然不能单独保证式 (12)、式 (13) 互为转置。这里讨论的是源码行为，本文未修改该实现。

**实现事实 2：一次调用只遍历一轮角度。**

[huo/art.py](https://github.com/zhiqwang/huo/blob/main/huo/art.py) 的 `art` 每次先执行 `img = torch.zeros(...)`，再用一次 `torch.randperm` 遍历角度，随后返回。`RadonFanbeam.art` 具有相同结构。

因此，定理 2 和定理 3 中的“重复扫描”尚未在这个函数内部发生。多次调用现有函数也不会延续上次估计，因为每次调用都会重新置零。即使某个理想化迭代满足渐近收敛条件，也不能由此推出当前一轮输出精确等于真解。

**实现事实 3：默认测量规模不足以唯一确定任意像素图像。**

按 [CLI 默认参数](https://github.com/zhiqwang/huo/blob/main/huo/cli.py)，图像为 $512\times512$，每角度 $500$ 个探测器，角度从 $0$ 到 $359$ 度。堆叠全部角度得到矩阵

$$
\mathbf A\in\mathbb R^{180000\times262144}.
$$

由秩与零度定理，

$$
\dim\ker\mathbf A
=262144-\operatorname{rank}(\mathbf A)
\geq262144-180000
=82144.
\tag{30}
$$

因此，若允许图像的所有像素独立变化，仅凭这些测量无法唯一确定任意图像。非负约束可能帮助某些具有特殊结构的图像，但不能普遍消除这个问题：若 $x_*$ 所有分量都严格为正，取任意非零 $z\in\ker\mathbf A$，总可以选足够小的 $\varepsilon\ne0$，使

$$
x_*+\varepsilon z\geq0,
\qquad
\mathbf A(x_*+\varepsilon z)=\mathbf A x_*.
$$

这同时给出了两个不同的非负图像和完全相同的投影。要主张唯一恢复，需要说明额外的支撑、结构或其他先验，而不能只引用 ART 收敛。

**实现事实 4：现有伴随测试没有验证伴随等式。**

[tests/test_art.py](https://github.com/zhiqwang/huo/blob/main/tests/test_art.py) 中的 `test_adjoint_property` 使用非负随机 $x,y$，最终检查两个内积有限且同号。由于正向、反向插值权重非负，这种同号现象本身不能证明

$$
\langle A_\theta x,y\rangle
=\langle x,B_\theta y\rangle.
\tag{31}
$$

欧氏伴随应检验式 (31) 的数值接近程度，例如使用包含正负值的随机向量，并控制相对于向量和算子输出尺度的误差。如果模型使用预先确定的加权内积

$$
\langle x,z\rangle_X=x^T H_Xz,
\qquad
\langle y,w\rangle_Y=y^T H_Yw,
$$

则相应条件是 $H_XB_\theta=A_\theta^TH_Y$，不能把欧氏内积下的差异直接解释为已满足加权伴随关系。

伴随测试只是核对算子关系，也不能单独证明选定步长和角度调度下的收敛。针对一个具体的小规模配置，可以通过对图像基向量和探测器基向量分别调用正反投影，构造实际的 $A_\theta,B_\theta$，再检验相应矩阵条件。这样的实验验证的是该配置；没有参数范围上的误差界或解析论证时，不能自动推广到其他分辨率和几何。
