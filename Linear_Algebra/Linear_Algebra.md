# 第 1 章 行列式
<span style="color:red">线性代数的定义顺序主要是因为能够引出后续的定义，教材的编写习惯总是连续的，前面的内容引出后面的内容，在查找时也可以当作方便的手册，并不会有一本完美的教材能够让人无痛接受数学，数学本就是一门需要多思考多总结，长期培养形成数学思维的学科，我觉得没必要总是要求别人发明自学式教材，这一块关于行列式的内容更多的是代数的维度在解释行列式，其他部分也一样，数学教材往往明规则只定义代数计算，而几何的角度往往是需要自己悟的，是另一套潜规则，我觉得关于线性代数的几何意义上的理解还是统一放在最后一部分比较好，因为几何意义往往需要多个定义放在一起理解，是多个知识点拼凑在一起的内容，可能这也是教材往往解释不清几何意义的原因吧</span>  
在线性代数中，最好以代数和几何两个维度去理解线代的理论，它是一种阐释多维度的非常好用的工具  
一方面，在代数上有一套自己的运算法则，在深度学习等大计算量的并行计算中应用广泛  
另一方面，在几何上有一套成熟的理论来解释几何的多维空间，并能方便的实现坐标系的变换  
### 核心思想：代数是“如何算”，几何是“是什么”
线性代数的精髓就在于：
*   它用**代数**的语言（矩阵、方程组）去精确地描述和操作**几何**的对象（空间、变换）。
*   它让我们能够通过**计算**来**想象**和理解高维空间中发生的事情。

---
### 在几何方面的意义：为空间提供坐标和变换
从几何的角度看，线性代数是描述和研究高维空间的“语言”。
1.  **坐标化空间**：向量 `v = (x, y, z)` 不仅仅是数字列表，它更是三维空间中一个点的**坐标**。线性代数将抽象的几何点、线、面，变成了可以具体计算的数字和数组。
2.  **描述变换**：这是最核心的一点。矩阵 `A` 代表了一个**几何变换**。比如：
    *   **旋转矩阵**：让整个空间绕某个轴旋转。
    *   **缩放矩阵**：让空间在某个方向上拉伸或压缩。
    *   **投影矩阵**：把三维空间“压”成一个二维平面。
    *   **剪切矩阵**：像推倒一摞书一样，让空间发生倾斜。
    当你计算 `Av = b` 时，在几何上，你是在看**向量 `v` 经过变换 `A` 后，变成了什么样子 `b`**。
3.  **描述几何对象**：
    *   **列空间**：一个矩阵所有列向量张成的空间，就是这个矩阵能“到达”的所有区域。
    *   **零空间**：所有能被矩阵“压扁”成零向量的向量集合，它描述了变换中“丢失”了哪些维度。
    *   **行列式**：一个标量，其绝对值代表了变换对**体积（或面积、长度）的缩放比例**。如果行列式为0，说明这个变换把空间“压扁”了，降维了。
### 在代数方面的意义：提供计算工具和抽象结构
从代数的角度看，线性代数为处理这些几何问题提供了强大的、系统化的计算工具。
1.  **系统化的计算工具**：解方程组 `Ax = b` 是最典型的代数问题。线性代数提供了一整套方法（高斯消元、矩阵求逆等）来解决这个问题。这个代数操作的几何意义就是：**找到一个点 `x`，它在经过变换 `A` 后，正好落在点 `b` 的位置上**。
2.  **抽象与统一**：代数擅长把具体问题抽象化。线性代数把各种几何变换（旋转、缩放等）都统一成**矩阵乘法**这一种代数运算。这意味着，复杂的几何问题可以转化为简单、规范的代数计算，这正是计算机能够处理图形学的基础。
3.  **分析结构**：代数工具能揭示几何变换的深层性质。例如，通过**特征值和特征向量**，我们可以找到一个变换中最“特殊”的方向：**特征向量**是在变换中只被拉伸或压缩，而方向保持不变的向量；**特征值**就是那个拉伸或压缩的比例。这在几何上意味着找到了变换的“主轴”。


---
## 1、二阶、三阶、n阶行列式
### 1.1 二元线性方程组与二阶行列式
对于二元线性方程组  
$$\begin{cases}
a_{11}x_1 + a_{12}x_2 = b_1 \\
a_{21}x_1 + a_{22}x_2 = b_2
\end{cases}
$$
将未知数的参数取出组成表达式$a_{11}a_{22}-a_{12}a_{21}$，成为参数表的二阶行列式，记作  
$$\begin{vmatrix}
a_{11} && a_{12} \\
\\
a_{21} && a_{22} 
\end{vmatrix}
$$
### 1.2 三阶行列式
类似二阶行列式，当有9个数构成$3 \times 3$的数表时，记三阶行列式为  
$$\begin{vmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33} \\
\end{vmatrix}
=
a_{11}a_{22}a_{33}+a_{12}a_{23}a_{31}+a_{13}a_{32}a_{21}-a_{13}a_{22}a_{31}-a_{11}a_{32}a_{23}-a_{12}a_{21}a_{33}
$$
### n阶行列式
根据逆序数和奇偶排列可以得到  
n阶行列式：  
$$D = 
\begin{vmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots &        & \vdots \\
a_{11} & a_{12} & \cdots & a_{1n} \\
\end{vmatrix}
=
\sum (-1)^ta_{1p_1}a_{2p_2}\cdots a_{np_n}
$$
## 2、行列式的性质
### 性质1 行列式与它的转置行列式相等
$$\begin{align*}
D &= det(a_{ij}) \\
D^T &= det(a_{ji}) \\
D &= D^T
\end{align*}
$$
### 性质2 对换行列式的两行(列)，行列式变号
显然，兑换两行或两列，改变所有排列的奇偶性，进而导致变号
$$\begin{gather*}
if : c_i \leftrightarrow c_j \lor r_i \leftrightarrow r_j \\
D^{'} = -D
\end{gather*}
$$
**推论**：如果行列式有两行(列)完全相同，则此行列式等于0  
$$\begin{gather*}
c_i (r_i) \leftrightarrow c_j (r_j) \\
D = -D \\
D = 0
\end{gather*}
$$
### 性质3 行列式的某一行(列)中所有的元素都乘同一元素$k$，等于用数$k$乘此行列式
显然，根据n阶行列式定义，合并同类项可得
$$\begin{align*}
\begin{vmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
\vdots & \vdots &        & \vdots \\
ka_{i1} & ka_{i2} & \cdots & ka_{in} \\
\vdots & \vdots &        & \vdots \\
a_{11} & a_{12} & \cdots & a_{1n} \\
\end{vmatrix}
&=
\sum (-1)^ta_{1p_1}\cdots ka_{ip_i}\cdots a_{np_n} \\
&=
k\sum (-1)^ta_{1p_1}\cdots a_{ip_i}\cdots a_{np_n} \\
\\
&= kD
\end{align*}
$$
### 性质4 行列式中如果有两行(列)元素成比例，则此行列式等于0
显然，由性质3和性质2的推论可以得到性质4
### 性质5 行列式的某一行(列)的元素都是两数之和，则D等于两个子行列式之和
说明行列式可拆分  
$$
D = 
\begin{vmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
\vdots & \vdots &        & \vdots \\
a_{i1}+a_{i1}^{'} & a_{i2}+a_{i2}^{'} & \cdots & a_{in}+a_{in}^{'} \\
\vdots & \vdots &        & \vdots \\
a_{11} & a_{12} & \cdots & a_{1n} \\
\end{vmatrix}
=
\begin{vmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
\vdots & \vdots &        & \vdots \\
a_{i1} & a_{i2} & \cdots & a_{in} \\
\vdots & \vdots &        & \vdots \\
a_{11} & a_{12} & \cdots & a_{1n} \\
\end{vmatrix}
+
\begin{vmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
\vdots & \vdots &        & \vdots \\
a_{i1}^{'} & a_{i2}^{'} & \cdots & a_{in}^{'} \\
\vdots & \vdots &        & \vdots \\
a_{11} & a_{12} & \cdots & a_{1n} \\
\end{vmatrix}
$$
### 性质6 把行列式的某一行(列)的各元素乘同一个数加到另一行(列)对应元素上，行列式不变
由性质5，和性质2推论可得  
$$
D = 
\begin{vmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
\vdots & \vdots &        & \vdots \\
a_{i1} & a_{i2} & \cdots & a_{in} \\
\vdots & \vdots &        & \vdots \\
a_{j1} & a_{j2} & \cdots & a_{jn} \\
\vdots & \vdots &        & \vdots \\
a_{11} & a_{12} & \cdots & a_{1n} \\
\end{vmatrix}
=
\begin{vmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
\vdots & \vdots &        & \vdots \\
a_{i1} & a_{i2} & \cdots & a_{in} \\
\vdots & \vdots &        & \vdots \\
a_{j1}+ka_{i1} & a_{j2}+ka_{i2} & \cdots & a_{jn}+ka_{in} \\
\vdots & \vdots &        & \vdots \\
a_{11} & a_{12} & \cdots & a_{1n} \\
\end{vmatrix}
(i \ne j)
$$
## 3、行列式按行(列)展开
### 余子式
n阶行列式中，将$a_{ij}$所在的行和列划去后，留下的行列式叫做$a_{ij}$的余子式，记作$M_{ij}$  
### 代数余子式  
余子式+上判断符号的前缀就是代数余子式，记
$$ A_{ij} = (-1)^{i+j}M_{ij} $$
### 定理2 行列式等于它任一行(列)的各元素与其对应的代数余子式成绩之和
$$\begin{gather*}
D = a_{i1}A_{i1}+a_{i2}A_{i2}+\cdots+a_{in}A_{in} & (i=1,2,\cdots, n)\\
D = a_{1j}A_{1j}+a_{2j}A_{2j}+\cdots+a_{nj}A_{nj} & (j=1,2,\cdots, n)
\end{gather*}
$$
### 推论 行列式某一行(列)的元素与另一行(列)的对应元素的代数余子式乘积之和等于0
形式上相当于有两行元素相同，根据行列式性质2的推论可以得到，即  
$$\begin{gather*}
D = a_{i1}A_{j1}+a_{i2}A_{j2}+\cdots+a_{in}A_{jn} & (i \ne j)\\
D = a_{1i}A_{1j}+a_{2i}A_{2j}+\cdots+a_{ni}A_{nj} & (i \ne j)
\end{gather*}
$$


# 第 2 章 矩阵及其运算
## 1、线性方程组
非齐次线性方程组
$$\begin{cases}
a_{11}x_1+a_{12}x_2+a_{13}x_3+\cdots+a_{1n}x_n = b_1 \\
a_{21}x_1+a_{22}x_2+a_{23}x_3+\cdots+a_{2n}x_n = b_2 \\
.......................... \\
a_{m1}x_1+a_{m2}x_2+a_{m3}x_3+\cdots+a_{mn}x_n = b_m \\
\end{cases}
$$
齐次线性方程组  
$$\begin{cases}
a_{11}x_1+a_{12}x_2+a_{13}x_3+\cdots+a_{1n}x_n = 0 \\
a_{21}x_1+a_{22}x_2+a_{23}x_3+\cdots+a_{2n}x_n = 0 \\
.......................... \\
a_{m1}x_1+a_{m2}x_2+a_{m3}x_3+\cdots+a_{mn}x_n = 0 \\
\end{cases}
$$
对于齐次线性方程组而言，一定有零解，但不一定有非零解  
判断方程组是否有解可以根据系数行列式$D \ne 0$来判断  
对于线性方程组需要讨论以下问题  
1. 线性方程组是否有解
2. 若有解，解是否唯一
3. 若存在多个解，如何求出多个解
## 2、矩阵的定义
### 定义1 由$m \times n$个数$a_{ij},(i=1,2,\cdots,m; j=1,2,\cdots,n)$排成m行n列的数表，称为$m \times n$矩阵，记作
$$
\boldsymbol{A} = 
\begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots &        & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn} \\
\end{pmatrix}
$$
有时在计算机中，为了显示的更加清楚也会采取方括号来表示矩阵  
$$
\boldsymbol{A} = 
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots &        & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn} \\
\end{bmatrix}
$$
* 元素是实数的矩阵称作实矩阵，复数称作复矩阵
* 行数列数相等的矩阵称作n阶方阵
* 只有一行的矩阵称作行矩阵，又称行向量
* 只有一列的矩阵称作列矩阵，又称列向量
* 两个矩阵行数列数分别相等称同型矩阵
* 若矩阵$\boldsymbol{A}$和$\boldsymbol{B}$是同型矩阵且对应元素相等，则矩阵$\boldsymbol{A}$和矩阵$\boldsymbol{B}$相等，即
$$\begin{gather*}
\boldsymbol{A} = (a_{ij})\\
\boldsymbol{B} = (b_ij)\\
a_{ij} = b_{ij} & (i=1,2,\cdots,m; j=1,2,\cdots,n)\\
\boldsymbol{A} = \boldsymbol{B}
\end{gather*} 
$$
* 元素都为$0$的矩阵称为零矩阵，记作$\boldsymbol{0}$，不同型的零矩阵是不同的   
* 如下几个有用的矩阵，其中$\boldsymbol{A}$称为系数矩阵，$\boldsymbol{x}$称为未知数矩阵，$\boldsymbol{b}$称为常数项矩阵，$\boldsymbol{B}$称为增广矩阵  
$$\begin{cases}
a_{11}x_1+a_{12}x_2+a_{13}x_3+\cdots+a_{1n}x_n = b_1 \\
a_{21}x_1+a_{22}x_2+a_{23}x_3+\cdots+a_{2n}x_n = b_2 \\
.......................... \\
a_{m1}x_1+a_{m2}x_2+a_{m3}x_3+\cdots+a_{mn}x_n = b_m \\
\end{cases}
$$
$$\begin{gather*}
\boldsymbol{A}=(a_{ij}),
\boldsymbol{x}=
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n \\
\end{pmatrix},
\boldsymbol{b}=
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_m \\
\end{pmatrix},
\boldsymbol{B}=
\begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} & b_1 \\
a_{21} & a_{22} & \cdots & a_{2n} & b_2 \\
\vdots & \vdots &        & \vdots & \vdots & \\
a_{m1} & a_{m2} & \cdots & a_{mn} & b_m \\
\end{pmatrix}
\end{gather*} 
$$
* 下面的关系表示从一个变量$x_1,x_2,\cdots,x_n$到另一个变量$y_1,y_2,\cdots,y_m$的线性变换
$$\begin{cases}
y_1 = a_{11}x_1+a_{12}x_2+a_{13}x_3+\cdots+a_{1n}x_n \\
y_2 = a_{21}x_1+a_{22}x_2+a_{23}x_3+\cdots+a_{2n}x_n \\
.......................... \\
y_m = a_{m1}x_1+a_{m2}x_2+a_{m3}x_3+\cdots+a_{mn}x_n \\
\end{cases}
$$
* 除了左上角到右下角对角线上以外的元素都为0，这种方阵称为对角阵，记作
$$
\boldsymbol{\Lambda} = diag(\lambda_1,\lambda_2,\cdots,\lambda_n) =
\begin{pmatrix}
\lambda_1&    0    & \cdots & 0 \\ 
    0    &\lambda_2& \cdots & 0 \\ 
  \vdots & \vdots  &        & \vdots \\
    0    &    0    & \cdots & \lambda_n \\ 
\end{pmatrix}
$$
* 特别的，当$\lambda_1=\lambda_2=\cdots=\lambda_n=1$时的线性变换称作恒等变换，对应的n阶方阵叫做n阶单位矩阵，简称单位阵
$$
\boldsymbol{I}=
\begin{pmatrix}
    1   &    0  & \cdots & 0 \\ 
    0   &   1   & \cdots & 0 \\ 
 \vdots & \vdots&        & \vdots \\
    0   &    0  & \cdots & 1 \\ 
\end{pmatrix}
$$
## 矩阵的运算
### 定义2 设有两个$m \times n$矩阵$\boldsymbol{A}=(a_{ij})$和$\boldsymbol{B}=(b_{ij})$，那么矩阵$\boldsymbol{A}$与$\boldsymbol{B}$的和记作$\boldsymbol{A}+\boldsymbol{B}$，规定
$$
\boldsymbol{A}+\boldsymbol{B}=
\begin{pmatrix}
a_{11}+b_{11} & a_{12}+b_{12} & \cdots & a_{1n}+b_{1n} \\
a_{21}+b_{21} & a_{22}+b_{22} & \cdots & a_{2n}+b_{2n} \\
\vdots        & \vdots        &        & \vdots        \\
a_{m1}+b_{m1} & a_{m2}+b_{m2} & \cdots & a_{mn}+b_{mn} \\
\end{pmatrix}
$$
注意: 只有当两个矩阵是同型矩阵时，这两个矩阵才能进行加法运算  
矩阵加法满足一下运算规律(设$\boldsymbol{A},\boldsymbol{B},\boldsymbol{C}$是$m \times n$矩阵)  
$$\begin{gather*}
(i) & \boldsymbol{A}+\boldsymbol{B} = \boldsymbol{B}+\boldsymbol{A} \\
(ii)& (\boldsymbol{A}+\boldsymbol{B})+\boldsymbol{C} = \boldsymbol{A}+(\boldsymbol{B}+\boldsymbol{C})
\end{gather*}
$$
设矩阵$\boldsymbol{A}=(a_{ij})$，记
$$
\boldsymbol{-A} = (-a_{ij})
$$
$\boldsymbol{-A}$称为$\boldsymbol{A}$的负矩阵，显然有
$$
\boldsymbol{A}+(\boldsymbol{-A}) = \boldsymbol{0}
$$
所以可以得到矩阵的减法
$$
\boldsymbol{A}-\boldsymbol{B} = \boldsymbol{A}+(\boldsymbol{-B})
$$
### 定义3 数$\lambda$与矩阵$\boldsymbol{A}$的乘积记作$\lambda\boldsymbol{A}$或$\boldsymbol{A}\lambda$，规定为
$$
\lambda\boldsymbol{A} = \boldsymbol{A}\lambda = 
\begin{pmatrix}
\lambda a_{11} & \lambda a_{12} & \cdots & \lambda a_{1n} \\
\lambda a_{21} & \lambda a_{22} & \cdots & \lambda a_{2n} \\
\vdots & \vdots &        & \vdots \\
\lambda a_{m1} & \lambda a_{m2} & \cdots & \lambda a_{mn} \\
\end{pmatrix}
$$
数乘矩阵满足下列运算规律(设$\boldsymbol{A},\boldsymbol{B}$为$m \times n$矩阵，$\lambda,\mu$为数):
$$\begin{gather*}
(i)  & (\lambda\mu)\boldsymbol{A} = \lambda(\mu\boldsymbol{A}) \\
(ii) & (\lambda+\mu)\boldsymbol{A} = \lambda\boldsymbol{A}+\mu\boldsymbol{A} \\
(iii)& \lambda(\boldsymbol{A}+\boldsymbol{B}) = \lambda\boldsymbol{A}+\lambda\boldsymbol{B}
\end{gather*}
$$
矩阵的加法与数乘统称矩阵的线性运算
### 定义4 设$\boldsymbol{A}=(a_{ij})$是一个$m \times s$矩阵，$\boldsymbol{B}=(b_{ij})$是一个$s \times n$矩阵，那么规定矩阵$\boldsymbol{A}$与矩阵$\boldsymbol{B}$的乘积是一个$m \times n$矩阵$\boldsymbol{C}=(c_{ij})$，其中
$$
c_{ij} = \sum_{k=1}^s a_{ik}b_{kj}
$$
按照定义，$1 \times s$的行矩阵与$s \times 1$的列矩阵乘积是一个数  
矩阵的乘积就是$\boldsymbol{A}\boldsymbol{B}=\boldsymbol{C}$的第$c_{ij}$个元素就是$\boldsymbol{A}$矩阵的$i$行与$\boldsymbol{B}$矩阵的$j$列的乘积  
矩阵的乘法不满足交换律，但是满足结合律和分配律
$$\begin{gather*}
(i)  & (\boldsymbol{A}\boldsymbol{B})\boldsymbol{C} = \boldsymbol{A}(\boldsymbol{B}\boldsymbol{C}) \\
(ii) & \lambda(\boldsymbol{A}\boldsymbol{B}) = (\lambda\boldsymbol{A})\boldsymbol{B}=\boldsymbol{A}(\lambda\boldsymbol{B}) \\
(iii)& \boldsymbol{A}(\boldsymbol{B}+\boldsymbol{C}) = \boldsymbol{A}\boldsymbol{B}+\boldsymbol{A}\boldsymbol{C},(\boldsymbol{B}+\boldsymbol{C})\boldsymbol{A}=\boldsymbol{B}\boldsymbol{A}+\boldsymbol{C}\boldsymbol{A}
\end{gather*}
$$
对于单位阵$\boldsymbol{I}$，显然有
$$
\boldsymbol{I_m}\boldsymbol{A_{m \times n}}=\boldsymbol{A_{m \times n}} \\
\boldsymbol{A_{m \times n}}\boldsymbol{I_n}=\boldsymbol{A_{m \times n}}
$$
即
$$
\boldsymbol{I}\boldsymbol{A} = \boldsymbol{A}\boldsymbol{I}=\boldsymbol{A}
$$
由于这个性质，所以有纯量阵
$$
\lambda\boldsymbol{I}=
\begin{pmatrix}
\lambda&&& \\
&\lambda&& \\
&&\ddots&  \\
&&&\lambda \\
\end{pmatrix}
$$
显然
$$
(\lambda\boldsymbol{I})\boldsymbol{A} = \lambda\boldsymbol{A} \\
\boldsymbol{A}(\lambda\boldsymbol{I}) = \lambda\boldsymbol{A}
$$
矩阵的幂，设$\boldsymbol{A}$是$n$阶方阵，则有矩阵的幂
$$
\boldsymbol{A^1}=\boldsymbol{A},\boldsymbol{A^{k+1}}=\boldsymbol{A^k}\boldsymbol{A^1}
$$
根据矩阵乘法结合律  
$$
\boldsymbol{A^k}\boldsymbol{A^l} = \boldsymbol{A_{k+l}} \\
(\boldsymbol{A^k})^l = \boldsymbol{A^{kl}}
$$
由上可知，线性变换可以写为
$$\begin{cases}
y_1 = a_{11}x_1+a_{12}x_2+a_{13}x_3+\cdots+a_{1n}x_n \\
y_2 = a_{21}x_1+a_{22}x_2+a_{23}x_3+\cdots+a_{2n}x_n \\
.......................... \\
y_m = a_{m1}x_1+a_{m2}x_2+a_{m3}x_3+\cdots+a_{mn}x_n \\
\end{cases}
$$
$$
\boldsymbol{y} = \boldsymbol{A}\boldsymbol{x}
$$
其中
$$
\boldsymbol{A} = (a_{ij}),
\boldsymbol{x} = 
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n \\
\end{pmatrix},
\boldsymbol{y} = 
\begin{pmatrix}
y_1 \\
y_2 \\
\vdots \\
y_m \\
\end{pmatrix}
$$
### 定义5 把矩阵$\boldsymbol{A}$的同序数行列互换得到的新矩阵，叫做$\boldsymbol{A}$的转置矩阵，记作$\boldsymbol{A}^T$
矩阵的转置满足一下运算规律
$$\begin{gather*}
(i)  & (\boldsymbol{A}^T)^T = \boldsymbol{A} \\
(ii) & (\boldsymbol{A}+\boldsymbol{B})^T = \boldsymbol{A}^T+\boldsymbol{B}^T \\
(iii)& (\lambda\boldsymbol{A})^T = \lambda\boldsymbol{A}^T \\
(iv) & (\boldsymbol{A}\boldsymbol{B})^T = \boldsymbol{B}^T\boldsymbol{A}^T 
\end{gather*}
$$
若满足$\boldsymbol{A}^T = \boldsymbol{A}$，称$\boldsymbol{A}$为对称阵
### 定义6 由$n$阶方阵$\boldsymbol{A}$的元素所构成的行列式(位置不变)，称为方阵$\boldsymbol{A}$的行列式，记作$det\boldsymbol{A}$或者$\begin{vmatrix}\boldsymbol{A}\end{vmatrix}$
矩阵行列式的运算规律
$$\begin{gather*}
(i)  & \begin{vmatrix}\boldsymbol{A}^T\end{vmatrix} = \begin{vmatrix}\boldsymbol{A}\end{vmatrix} \\
(ii) & \begin{vmatrix}\lambda\boldsymbol{A}\end{vmatrix} = \lambda^n\begin{vmatrix}\boldsymbol{A}\end{vmatrix} \\
(iii)& \begin{vmatrix}\boldsymbol{A}\boldsymbol{B}\end{vmatrix} = \begin{vmatrix}\boldsymbol{A}\end{vmatrix}\begin{vmatrix}\boldsymbol{B}\end{vmatrix} \\
\end{gather*}
$$
### 定义7 对于$n$阶矩阵$\boldsymbol{A}$，如果有一个$n$矩阵$\boldsymbol{B}$，使得$\boldsymbol{A}\boldsymbol{B} = \boldsymbol{B}\boldsymbol{A} = \boldsymbol{I}$，就说矩阵$\boldsymbol{A}$是可逆的，并且将$\boldsymbol{B}$称为$\boldsymbol{A}$的逆矩阵，简称逆阵，记作$\boldsymbol{A}^{-1}$
$$
\boldsymbol{A}\boldsymbol{B} = \boldsymbol{B}\boldsymbol{A} = \boldsymbol{I} \\
\boldsymbol{B} = \boldsymbol{A}^{-1}
$$
如果矩阵$\boldsymbol{A}$是可逆的，那么$\boldsymbol{A}$的逆矩阵是惟一的，假设$\boldsymbol{B},\boldsymbol{C}$都是$\boldsymbol{A}$的逆矩阵，则有
$$
\boldsymbol{B} = \boldsymbol{B}\boldsymbol{I} = \boldsymbol{B}(\boldsymbol{A}\boldsymbol{C})
= (\boldsymbol{B}\boldsymbol{A})\boldsymbol{C} = \boldsymbol{I}\boldsymbol{C} = \boldsymbol{C}
$$
所以$\boldsymbol{A}$的逆矩阵是惟一的
### 定理1 若矩阵A可逆，则$\begin{vmatrix}\boldsymbol{A}\end{vmatrix}\ne 0$
$$
\boldsymbol{A}\boldsymbol{A}^{-1} = \boldsymbol{I} \\
\begin{vmatrix}\boldsymbol{A}\boldsymbol{A}^{-1}\end{vmatrix} = \begin{vmatrix}\boldsymbol{I}\end{vmatrix} \\
\begin{vmatrix}\boldsymbol{A}\end{vmatrix}\begin{vmatrix}\boldsymbol{A}^{-1}\end{vmatrix} = \begin{vmatrix}\boldsymbol{I}\end{vmatrix} = 1 \\
\begin{vmatrix}\boldsymbol{A}\end{vmatrix} \ne 0
$$
### 定理2 若$\begin{vmatrix}\boldsymbol{A}\end{vmatrix} \ne 0$，则矩阵$\boldsymbol{A}$可逆，且
$$
\boldsymbol{A}^{-1} = \frac{1}{\begin{vmatrix}\boldsymbol{A}\end{vmatrix}}\boldsymbol{A}^*
$$
其中，$\boldsymbol{A}^*$为伴随矩阵  
$$
\boldsymbol{A}^* = 
\begin{pmatrix}
A_{11}  &  A_{21}  &  \cdots  &  A_{n1} \\
A_{12}  &  A_{22}  &  \cdots  &  A_{n2} \\
\vdots  &  \vdots  &          &  \vdots \\
A_{1n}  &  A_{2n}  &  \cdots  &  A_{nn} \\
\end{pmatrix}
$$
当$\begin{vmatrix}\boldsymbol{A}\end{vmatrix} = 0$时，$\boldsymbol{A}$称为奇异矩阵，否则称为非奇异矩阵  
由定理1和定理2可知，$\boldsymbol{A}$是可逆矩阵的充分必要条件是$\begin{vmatrix}\boldsymbol{A}\end{vmatrix} \ne 0$  
### 由定理1和定理2，产生如下推论
* 若$\boldsymbol{A}$可逆，则$\boldsymbol{A}^{-1}$也可逆，且$(\boldsymbol{A}^{-1})^{-1} = \boldsymbol{A}$  
* 若$\boldsymbol{A}$可逆，数$\lambda \ne 0$，则$(\lambda\boldsymbol{A})^{-1} = \frac{1}{\lambda}\boldsymbol{A}^{-1}$  
* 若$\boldsymbol{A},\boldsymbol{B}$为同阶矩阵且均可逆，则$\boldsymbol{A}\boldsymbol{B}$也可逆，且$(\boldsymbol{A}\boldsymbol{B})^{-1} = \boldsymbol{B}^{-1}\boldsymbol{A}^{-1}$
* 若$\boldsymbol{A}$可逆，则$\boldsymbol{A}^T$可逆，且$(\boldsymbol{A}^T)^{-1} = (\boldsymbol{A}^{-1})^T$
### Cramer法则
对于线性方程组  
$$\begin{cases}
a_{11}x_1+a_{12}x_2+a_{13}x_3+\cdots+a_{1n}x_n = b_1 \\
a_{21}x_1+a_{22}x_2+a_{23}x_3+\cdots+a_{2n}x_n = b_2 \\
.......................... \\
a_{m1}x_1+a_{m2}x_2+a_{m3}x_3+\cdots+a_{mn}x_n = b_m \\
\end{cases}\\
$$
也就是矩阵方程
$$
\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}
$$
如果线性方程组的系数矩阵的行列式不等于0，即  
$$
\begin{vmatrix}\boldsymbol{A}\end{vmatrix} = 
\begin{vmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots &        & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn} \\
\end{vmatrix}
\ne 0
$$
则方程有惟一解
$$
\boldsymbol{x} = (x_1, x_2, \cdots, x_n)^T \\
x_i = \frac{\begin{vmatrix}\boldsymbol{A}_i\end{vmatrix}}{\begin{vmatrix}\boldsymbol{A}\end{vmatrix}},(i=1,2,\cdots,n)
$$
### 分块矩阵
内容太多，建议看书  
众多分块方法中，按行分块和按列分块两种方式应该得到重视  
设$m \times n$矩阵$\boldsymbol{A}=(a_{ij})$则矩阵$\boldsymbol{A}$有$n$个列向量，第j列记作
$$
\boldsymbol{a_j} = 
\begin{pmatrix}
a_{1j} & a_{2j} & \cdots & a_{mj}
\end{pmatrix}^T
$$
$\boldsymbol{A}$按列分块为
$$
\boldsymbol{A} = 
\begin{pmatrix}
\boldsymbol{a_1} & \boldsymbol{a_2} & \cdots & \boldsymbol{a_n}
\end{pmatrix}
$$
记第i行为
$$
\boldsymbol{\alpha_i} = 
\begin{pmatrix}
a_{i1} & a_{i2} & \cdots & a_{in}
\end{pmatrix}^T
$$
$\boldsymbol{A}$按行分块为
$$
\boldsymbol{A} = 
\begin{pmatrix}
\boldsymbol{a_1}^T \\ \boldsymbol{a_2}^T \\ \vdots \\ \boldsymbol{a_m}^T
\end{pmatrix}
$$
# 第3章 矩阵的初等变换与线性方程组
## 1、矩阵的初等变换
### 定义1 矩阵的初等行(列)变换
* 对换两行(列)，记作$r_i \leftrightarrow r_j,c_i \leftrightarrow c_j$
* 以数$k \ne 0$乘某行(列)的所有元，记作$r_i \times k, c_i \times k$
* 把某一行(列)的所有元的$k$倍加到另一行(列)对应的元上去，记作$r_i+kr_j, c_i+kc_j$  
---  
如果矩阵$\boldsymbol{A}$经过有限次初等行变换变成矩阵$\boldsymbol{B}$，则称$\boldsymbol{A}$与$\boldsymbol{B}$行等价，记作$\boldsymbol{A} \overset{r}{\sim} \boldsymbol{B}$  
如果矩阵$\boldsymbol{A}$经过有限次初等列变换变成矩阵$\boldsymbol{B}$，则称$\boldsymbol{A}$与$\boldsymbol{B}$列等价，记作$\boldsymbol{A} \overset{c}{\sim} \boldsymbol{B}$  
如果矩阵$\boldsymbol{A}$经过有限次初等变换变成矩阵$\boldsymbol{B}$，则称$\boldsymbol{A}$与$\boldsymbol{B}$等价，记作$\boldsymbol{A} \sim \boldsymbol{B}$  
矩阵间的等价关系具有以下性质  
* 反身性 $\boldsymbol{A} \sim \boldsymbol{A}$
* 对称性 $\boldsymbol{A} \sim \boldsymbol{B},\boldsymbol{B} \sim \boldsymbol{A}$
* 传递性 $\boldsymbol{A} \sim \boldsymbol{B},\boldsymbol{B} \sim \boldsymbol{C},\boldsymbol{A} \sim \boldsymbol{C}$  
### 定义2 行阶梯形矩阵 and 行最简形矩阵 and 标准形
一个矩阵的行最简形矩阵是惟一确定，对行最简形矩阵进行初等列变换，可以得到一种形状更简单的矩阵称为标准形  
标准形的特点是左上角为单位阵，其余元素为0，形如
$$
\boldsymbol{F} = 
\begin{pmatrix}
\boldsymbol{I}_r & \boldsymbol{0} \\
\boldsymbol{0}   & \boldsymbol{0}
\end{pmatrix}_{m \times n}
$$
### 定理1 矩阵的初等变换的性质
设$A,B$为$m \times n$的矩阵，则有  
* $\boldsymbol{A} \overset{r}{\sim} \boldsymbol{B}$的充分必要条件是存在$m$阶可逆矩阵$\boldsymbol{P}$，使得$\boldsymbol{P}\boldsymbol{A} = \boldsymbol{B}$
* $\boldsymbol{A} \overset{c}{\sim} \boldsymbol{B}$的充分必要条件是存在$n$阶可逆矩阵$\boldsymbol{Q}$，使得$\boldsymbol{A}\boldsymbol{Q} = \boldsymbol{B}$
* $\boldsymbol{A} \sim \boldsymbol{B}$的充分必要条件是存在$m$阶可逆矩阵$\boldsymbol{P}$，$n$阶可逆矩阵$\boldsymbol{Q}$，使得$\boldsymbol{P}\boldsymbol{A}\boldsymbol{Q} = \boldsymbol{B}$
### 定义3 由单位阵$\boldsymbol{I}$经过一次初等变换得到的矩阵称为初等矩阵，三种初等变换对应了三种初等矩阵
将三种初等矩阵左乘矩阵$\boldsymbol{A} = (a_{ij})_{m \times n}$，得到  
初等行变换1
$$
\boldsymbol{I}(i,j) = 
\begin{pmatrix}
1 & & & & & & & & & &\\
& \ddots & & & & & & & & &\\
& & 1 & & & & & & & &\\
& & & 0 & & & & 1 & & &\\
& & & & 1 & & & & & &\\
& & & & & \ddots & & & & &\\
& & & & & & 1 & & & &\\
& & & 1 & & & & 0 & & &\\
& & & & & & & & 1 & &\\
& & & & & & & & & \ddots &\\
& & & & & & & & & & 1\\
\end{pmatrix}
$$
$$
\boldsymbol{I}(i,j)\boldsymbol{A} = 
\begin{pmatrix}
a_{11} & a{12} & \cdots & a_{1n} \\
\vdots &\vdots &        & \vdots \\
a_{j1} & a{j2} & \cdots & a_{jn} \\
\vdots &\vdots &        & \vdots \\
a_{i1} & a{i2} & \cdots & a_{in} \\
\vdots &\vdots &        & \vdots \\
a_{n1} & a{n2} & \cdots & a_{nn}
\end{pmatrix}
$$
初等行变换2
$$
\boldsymbol{I}(i(k)) = 
\begin{pmatrix}
1 & & & & & & \\
& \ddots & & & & &\\
& & 1 & & & & \\
& & & k & & & \\
& & & & 1 & & \\
& & & & & \ddots & \\
& & & & & & 1 
\end{pmatrix}
$$
$$
\boldsymbol{I}(i(k))\boldsymbol{A} = 
\begin{pmatrix}
a_{11} & a{12} & \cdots & a_{1n} \\
\vdots &\vdots &        & \vdots \\
ka_{i1} & ka{i2} & \cdots & ka_{in} \\
\vdots &\vdots &        & \vdots \\
a_{n1} & a{n2} & \cdots & a_{nn}
\end{pmatrix}
$$
初等行变换3
$$
\boldsymbol{I}(i+j(k)) = 
\begin{pmatrix}
1 & & & & & & \\
& \ddots & & & & &\\
& & 1 & & k & & \\
& & & \ddots & & & \\
& & & & 1 & & \\
& & & & & \ddots & \\
& & & & & & 1 
\end{pmatrix}
$$
$$
\boldsymbol{I}(i+j(k))\boldsymbol{A} = 
\begin{pmatrix}
a_{11} & a{12} & \cdots & a_{1n} \\
\vdots &\vdots &        & \vdots \\
a_{i1}+ka_{j1} & a{i2}+ka_{j2} & \cdots & a_{in}+ka_{jn} \\
\vdots &\vdots &        & \vdots \\
a_{n1} & a{n2} & \cdots & a_{nn}
\end{pmatrix}
$$
### 性质1 设$\boldsymbol{A}$是一个$m\times n$矩阵，对$\boldsymbol{A}$实施一次初等行变换，相当于在$\boldsymbol{A}$左边乘相应的$m$阶初等矩阵；对$\boldsymbol{A}$实施一次初等列变换，相当于在$\boldsymbol{A}$右边乘相应的$n$阶初等矩阵
显然初等矩阵均可逆，且逆矩阵是同一类型的初等矩阵
$$
\boldsymbol{I}(i, j)^{-1} = \boldsymbol{I}(i, j)
$$
$$
\boldsymbol{I}(i(k))^{-1} = \boldsymbol{I}(i(\frac{1}{k}))
$$
$$
\boldsymbol{I}(i+j(k))^{-1} = \boldsymbol{I}(i+j(-k))
$$
### 性质2 方阵$\boldsymbol{A}$可逆的充分必要条件是存在有限个初等矩阵$\boldsymbol{P}_1,\boldsymbol{P}_2,\cdots,\boldsymbol{P}_l$，使得$\boldsymbol{A} = \boldsymbol{P}_1\boldsymbol{P}_2\cdots\boldsymbol{P}_l$
### 推论 对于方阵，方阵$\boldsymbol{A}$可逆的充分必要条件是$\boldsymbol{A} \overset{r}{\sim} \boldsymbol{I}$
### 几种增广矩阵
$$
\begin{pmatrix}
\boldsymbol{A} & \boldsymbol{I}
\end{pmatrix}
\sim
\begin{pmatrix}
\boldsymbol{I} & \boldsymbol{A}^{-1}
\end{pmatrix}
$$
$$
\boldsymbol{P}\boldsymbol{A} = \boldsymbol{B}
\Leftrightarrow
\begin{pmatrix}
\boldsymbol{A} & \boldsymbol{I}
\end{pmatrix}
\sim
\begin{pmatrix}
\boldsymbol{B} & \boldsymbol{P}
\end{pmatrix}
$$
$$
\boldsymbol{A}\boldsymbol{X} = \boldsymbol{Y}
\Leftrightarrow
\begin{pmatrix}
\boldsymbol{A} & \boldsymbol{Y}
\end{pmatrix}
\sim
\begin{pmatrix}
\boldsymbol{I} & \boldsymbol{X}
\end{pmatrix}
$$
$$
\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}
\Leftrightarrow
\begin{pmatrix}
\boldsymbol{A} & \boldsymbol{b}
\end{pmatrix}
\sim
\begin{pmatrix}
\boldsymbol{I} & \boldsymbol{x}
\end{pmatrix}
$$
## 2、矩阵的秩
### 定义4 在$m \times n$中，任取$k$行和与$k$列$(k \le m, k \le m)$，位于这些行列交叉处的$k^2$个元素，不改变它们在$\boldsymbol{A}$中所处的位置次序而得的$k$阶行列式，称为矩阵$\boldsymbol{A}$的$k$阶子式  
那么根据定义4，参考下方的矩阵$\boldsymbol{A}$  
$$
\boldsymbol{A} = 
\begin{pmatrix}
1 & 1 & -2 & 1 & 4\\
0 & 1 & -1 & 1 & 0\\
0 & 0 & 0 & 1 & -3\\
0 & 0 & 0 & 0 & 0\\
\end{pmatrix}
$$
取1、2、3行，1、2、4列得到三阶非零子式
$$
\begin{vmatrix}
1 & 1 & 1\\
0 & 1 & 1\\
0 & 0 & 1\\
\end{vmatrix}
$$
有这样一个问题，如果我在$\boldsymbol{A}$中任取4行4列，那么它的任一子式都将因为含有0行而值为0；也就是说，$\boldsymbol{A}$的非0子式最高阶数是3.  
### 引理 设$\boldsymbol{A} \overset{r}{\sim} \boldsymbol{B}$，则$\boldsymbol{A}$与$\boldsymbol{B}$中非0子式的最高阶数相等
证明还是比较 严谨，但是我觉得可以这样想，矩阵的行变换在代数角度上来说是线性方程组的组合与变换，无论你怎么变换方程组，总有些量是消不掉的，也就是最高阶数相等  
### 定义5 设在矩阵$\boldsymbol{A}$中有一个不等于0的$r$阶子式$\boldsymbol{D}$，且所有$r+1$阶子式(如果存在的话)全部等于0，那么$\boldsymbol{D}$称为矩阵$\boldsymbol{A}$的最高阶非0子式，数$r$称为矩阵$\boldsymbol{A}$的秩，记作$R(\boldsymbol{A})$，并规定零矩阵的秩等于0  
显然，若$\boldsymbol{A}$为$m \times n$矩阵，则
$$0 \le R(\boldsymbol{A}) \le \min\{m, n\}$$
由于行列式与转置行列式相等，则有
$$R(\boldsymbol{A}^T) = R(\boldsymbol{A})$$
对于$n$阶方阵$\boldsymbol{A}$  
$$
\begin{vmatrix}
\boldsymbol{A}
\end{vmatrix}
\ne 0, 
R(\boldsymbol{A}) = n
$$
$$
\begin{vmatrix}
\boldsymbol{A}
\end{vmatrix}
= 0, 
R(\boldsymbol{A}) < n
$$
可以初步看出，可逆矩阵的秩等于矩阵的阶数，不可逆矩阵的秩小于矩阵的阶数，因此，可逆矩阵又称**满秩矩阵**，不可逆矩阵又称**降秩矩阵**  
<span style="color:red">矩阵的秩是研究矩阵问题的核心，这里只是秩的一种定义，实际上秩的概念涵盖了代数和几何的很多个维度，既在代数角度代表了线性方程组的一些性质，又在集合维度隐含了空间和向量的一些问题，在后面的内容会对秩有一个总结，这里只需要注意多去联想秩的概念，秩可以将整个线性代数的体系联结起来</span>

### 定理2 若$\boldsymbol{A} \sim \boldsymbol{B}$，则$R(\boldsymbol{A}) = R(\boldsymbol{B})$
### 推论 若可逆矩阵$\boldsymbol{P},\boldsymbol{Q}$，使得$\boldsymbol{PAQ} = \boldsymbol{B}$，则$R(\boldsymbol{A}) = R(\boldsymbol{B})$  
不难看出，根据初等矩阵和初等变换的关系，以及初等矩阵的性质2，很容易可以得到推论$\boldsymbol{PAQ} = \boldsymbol{B}$  
根据定理2和推论，可以得到，矩阵的初等变换作为一种运算，它不改变矩阵的秩
### 矩阵秩的性质
* $0 \le R(\boldsymbol{A}) \le \min\{m, n\}$
* $R(\boldsymbol{A}^T) = R(\boldsymbol{A})$
* 若$\boldsymbol{A} \sim \boldsymbol{B}$，则$R(\boldsymbol{A}) = R(\boldsymbol{B})$
* 若矩阵$\boldsymbol{P},\boldsymbol{Q}$可逆，则$R(\boldsymbol{A}) = R(\boldsymbol{PAQ})$
* $\max\{R(\boldsymbol{A}),R(\boldsymbol{B})\} \le R(\boldsymbol{A},\boldsymbol{B}) \le R(\boldsymbol{A})+R(\boldsymbol{B})$，特别的，当$\boldsymbol{B}=\boldsymbol{b}$为非0列向量时，有$R(\boldsymbol{A}) \le R(\boldsymbol{A},\boldsymbol{b}) \le R(\boldsymbol{A})+1$  
* $R(\boldsymbol{A}+\boldsymbol{B}) \le R(\boldsymbol{A})+R(\boldsymbol{B})$
* $R(\boldsymbol{A}\boldsymbol{B}) \le \min\{R(\boldsymbol{A}),R(\boldsymbol{B})\}$
* 若$\boldsymbol{A}_{m \times n}\boldsymbol{B}_{n \times l} = \boldsymbol{0}$，则$R(\boldsymbol{A})+R(\boldsymbol{B}) \le n$
* 设$\boldsymbol{A}\boldsymbol{B} = \boldsymbol{0}$，若$\boldsymbol{A}$为列满秩矩阵，则$\boldsymbol{B} = \boldsymbol{0}$  
## 3、线性方程组的解
设有$n$个未知数$m$个方程的线性方程组
$$\begin{cases}
a_{11}x_1+a_{12}x_2+a_{13}x_3+\cdots+a_{1n}x_n = b_1 \\
a_{21}x_1+a_{22}x_2+a_{23}x_3+\cdots+a_{2n}x_n = b_2 \\
.......................... \\
a_{m1}x_1+a_{m2}x_2+a_{m3}x_3+\cdots+a_{mn}x_n = b_m \\
\end{cases}
$$
方程组可以写成以向量$x$为未知元的向量方程
$$
\boldsymbol{Ax} = \boldsymbol{b}
$$
如果这个方程有解，就称方程是相容的；如果无解就称是不相容的  
显然由上一小节，利用系数矩阵$\boldsymbol{A}$和增广矩阵$(\boldsymbol{A},\boldsymbol{b})$的秩，可以方便的讨论线性方程组是否有解(相容)，以及有解时是否惟一等问题  
### 定理3 $n$元线性方程组$\boldsymbol{Ax} = \boldsymbol{b}$
* 无解的充分必要条件是$R(\boldsymbol{A}) < R(\boldsymbol{A},\boldsymbol{b})$
* 有惟一解的充分必要条件是$R(\boldsymbol{A}) = R(\boldsymbol{A},\boldsymbol{b}) = n$
* 有无限多解的充分必要条件是$R(\boldsymbol{A}) = R(\boldsymbol{A},\boldsymbol{b}) < n$
### 定理4 $n$元齐次线性方程组$\boldsymbol{Ax}=\boldsymbol{0}$有非0解的充分必要条件是$R(\boldsymbol{A})<n$  
### 定理5 线性方程组$\boldsymbol{Ax}=\boldsymbol{b}$有解的充分必要条件时$R(\boldsymbol{A}) = R(\boldsymbol{A},\boldsymbol{b})$  
### 定理6 矩阵方程$\boldsymbol{AX} = \boldsymbol{B}$有解的充分必要条件是$R(\boldsymbol{A}) = R(\boldsymbol{A},\boldsymbol{B})$  
### 定理7 设$\boldsymbol{AB}=\boldsymbol{C}$，则$R(\boldsymbol{C}) \le \min\{R(\boldsymbol{A}),R(\boldsymbol{B})\}$

# 第4章 向量组的线性相关性
## 1、向量组及其线性组合
### 定义1 n个有次序的数$a_1, a_2, \cdots, a_n$所组成的数组称为$n$维向量，这$n$个数称为该向量的$n$个分量，第$i$个数$a_i$称为第$i$个分量
向量如果没有特殊说明，默认指列向量，向量的定义如下
$$
\boldsymbol{a} = 
\begin{pmatrix}
a_1 \\
a_2 \\
\vdots \\
a_n
\end{pmatrix},
\boldsymbol{a}^T = 
(a_1, a_2, \cdots, a_n)
$$
* $n$维向量的全体所组成的集合，称为$n$维向量空间
$$
\R^n  = 
\{\boldsymbol{x}=(x_1,x_2,\cdots,x_n)^T | x_1,x_2,\cdots,x_n \in \R \}
$$

* 若干个同维数的列向量(或同维数的行向量)所组成的集合叫做向量组，$m$个$n$维列向量所组成的向量组$A: \boldsymbol{a}_1,\boldsymbol{a}_2,\cdots,\boldsymbol{a}_m$构成一个$n \times m$矩阵  
$$
\boldsymbol{A} = (\boldsymbol{a}_1,\boldsymbol{a}_2,\cdots,\boldsymbol{a}_m)
$$
$m$个$n$维行向量所组成的向量组$B: \boldsymbol{a}_1^T,\boldsymbol{a}_2^T,\cdots,\boldsymbol{a}_m^T$构成一个$n \times m$矩阵  
$$
\boldsymbol{B} = 
\begin{pmatrix}
\boldsymbol{a}_1^T \\
\boldsymbol{a}_2^T \\
\vdots \\
\boldsymbol{a}_m^T
\end{pmatrix}
$$

### 定义2  
给定向量组$A: \boldsymbol{\alpha}_1,\boldsymbol{\alpha}_2,\cdots,\boldsymbol{\alpha}_m$，对于任何一组实数$k_1, k_2, \cdots, k_m$，表达式
$$
k_1\boldsymbol{\alpha}_1+k_2\boldsymbol{\alpha}_2+\cdots+k_m\boldsymbol{\alpha}_m
$$
称为向量组$A$的一个线性组合，$k_1, k_2, \cdots, k_m$称为这个线性组合的系数  
给定向量组$A: \boldsymbol{\alpha}_1,\boldsymbol{\alpha}_2,\cdots,\boldsymbol{\alpha}_m$和向量$\boldsymbol{b}$，对于任何一组实数$\lambda_1, \lambda_2, \cdots, \lambda_m$，使得
$$
\boldsymbol{b} = \lambda_1\boldsymbol{\alpha}_1+\lambda_2\boldsymbol{\alpha}_2+\cdots+\lambda_m\boldsymbol{\alpha}_m
$$
则称向量$\boldsymbol{b}$是向量组$A$的线性组合，这时称向量$\boldsymbol{b}$能由向量组$A$线性表示，即方程组  
$$
x_1\boldsymbol{\alpha}_1+x_2\boldsymbol{\alpha}_2+\cdots+x_m\boldsymbol{\alpha}_m = \boldsymbol{b}
$$
有解  
### 定理1 向量$\boldsymbol{b}$能由向量组$\boldsymbol{A} = (\boldsymbol{\alpha}_1,\boldsymbol{\alpha}_2,\cdots,\boldsymbol{\alpha}_m)$线性表示的充分必要条件是矩阵$R(\boldsymbol{A}) = R((\boldsymbol{A},\boldsymbol{b}))$  
### 定义3 设有两个向量组$A: \boldsymbol{a}_1,\boldsymbol{a}_2,\cdots,\boldsymbol{a}_m,B: \boldsymbol{b}_1,\boldsymbol{b}_2,\cdots,\boldsymbol{b}_l$，若$B$中每个向量都能由向量组$A$线性表示，则称向量组$B$能由向量组$A$线性表示，若向量组$A$与向量组$B$能相互线性表示，则称这两个向量组等价
### 定理2 向量组$B: \boldsymbol{b}_1,\boldsymbol{b}_2,\cdots,\boldsymbol{b}_l$能听由向量组$A: \boldsymbol{a}_1,\boldsymbol{a}_2,\cdots,\boldsymbol{a}_m$线性表示的充分必要条件是矩阵$\boldsymbol{A}$的秩等于矩阵$\boldsymbol{(A,B)}$的秩，即$R(\boldsymbol{A}) = R(\boldsymbol{(A,B)})$
### 定理3 设向量组$B: \boldsymbol{b}_1,\boldsymbol{b}_2,\cdots,\boldsymbol{b}_l$能听由向量组$A: \boldsymbol{a}_1,\boldsymbol{a}_2,\cdots,\boldsymbol{a}_m$线性表示，则有$R(\boldsymbol{B}) \le R(\boldsymbol{A})$  
实际上就是高维可以表示低维，但是低维不能表示高维
## 2、向量组的线性相关性
### 定义4 
给定向量组$A: \boldsymbol{\alpha}_1,\boldsymbol{\alpha}_2,\cdots,\boldsymbol{\alpha}_m$；如果存在不全为0的数$k_1, k_2, \cdots, k_m$，使
$$
k_1\boldsymbol{\alpha}_1+k_2\boldsymbol{\alpha}_2+\cdots+k_m\boldsymbol{\alpha}_m = \boldsymbol{0}
$$
则称向量组$A$线性相关，否则称向量组$A$线性无关($k_1=k_2=\cdots=k_m=0$)

---  
结合定义4可以发现，当向量组$A: \boldsymbol{\alpha}_1,\boldsymbol{\alpha}_2,\cdots,\boldsymbol{\alpha}_m$线性相关时，存在至少一个向量的系数$k \ne 0$  
$$
k_1\boldsymbol{\alpha}_1+\cdots+k_i\boldsymbol{\alpha}_i+\cdots+k_m\boldsymbol{\alpha}_m = \boldsymbol{0}
$$
$$
\boldsymbol{\alpha}_i = \frac{-1}{k_i}(k_1\boldsymbol{\alpha}_1+k_2\boldsymbol{\alpha}_2+\cdots+k_m\boldsymbol{\alpha}_m)
$$
即，有$\lambda_1,\lambda_2,\cdots,\lambda_{m-1}$使得
$$
\boldsymbol{\alpha}_i = \lambda_1\boldsymbol{\alpha}_1+\cdots+\lambda_i\boldsymbol{\alpha}_{i+1}+\cdots+\lambda_{m-1}\boldsymbol{\alpha}_m
$$
这样就把线性相关和方程组联系在了一起，当方程组$\boldsymbol{Ax} = \boldsymbol{b}$或$\boldsymbol{Ax} = \boldsymbol{0}$时，$\boldsymbol{x}$有非零解的充分必要条件只需要对应的参数矩阵$\boldsymbol{A},(\boldsymbol{A,b})$线性相关

### 定理4 由$m$个$n$维向量构成的向量组$A: \alpha_1, \alpha_2, \cdots, \alpha_m$线性相关的充分必要条件时她所构成的矩阵$\boldsymbol{A} = (\boldsymbol{\alpha_1},\boldsymbol{\alpha_2},\cdots,\boldsymbol{\alpha_m})$的秩$R(\boldsymbol{A}) \le m$；向量组$A$线性无关的充分必要条件是$R(\boldsymbol{A}) = m$

### 定理5
### (1) 若向量组 $A: \boldsymbol{\alpha_1},\boldsymbol{\alpha_2},\cdots,\boldsymbol{\alpha_m}$ 线性相关，则向量组 $B: \boldsymbol{\alpha_1},\boldsymbol{\alpha_2},\cdots,\boldsymbol{\alpha_m},\boldsymbol{\alpha_{m+1}}$ 也线性相关；反之，若 $B$ 线性无关，$A$ 也线性无关
### (2) $m$ 个 $n$ 维向量组成的向量组，当维数 $n$ 小于向量个数 $m$ 时一定线性相关，特别的$n+1$个$n$维向量一定线性相关
### (3) 设向量组 $A: \alpha_1, \alpha_2, \cdots, \alpha_m$ 线性无关，而向量组 $B: \alpha_1, \alpha_2, \cdots, \alpha_m, b$ 线性相关，则向量 $b$ 一定能用向量组$A$来线性表示，且表达式唯一

## 3、向量组的秩
在这节往后，开始讨论含有无限多个向量的向量组
### 定义5 设有向量组$A$ ，如果在 $A$ 中能选出 $r$ 个向量 $\boldsymbol{\alpha_1},\boldsymbol{\alpha_2},\cdots,\boldsymbol{\alpha_r}$，满足
* 向量组 $A_0: \boldsymbol{\alpha_1},\boldsymbol{\alpha_2},\cdots,\boldsymbol{\alpha_r}$ 线性无关；
* 向量组 $A$ 中任意 $r+1$ 个向量(如果有)都线性相关；
### 那么称向量组 $A_0$ 是向量组 $A$ 的一个最大线性无关向量组(简称最大无关组)，最大无关组所含的向量个数 $r$ 称为向量组 $A$ 的秩，记作 $R_A$ ，特别的，规定零向量的秩为 $0$

### 推论(最大线性无关组的等价定义) 设向量组 $A_0: \boldsymbol{\alpha_1},\boldsymbol{\alpha_2},\cdots,\boldsymbol{\alpha_r}$ 是向量组 $A$ 的一个部分组，且满足
* 向量组 $A_0$ 线性无关
* 向量组 $A$ 的任意一个向量都能由 $A_0$ 线性表示
### 那么向量组 $A_0$ 便是向量组 $A$ 的一个最大线性无关组
对于 $n$ 维向量空间 $\R^n$ 有很多最大线性无关组，任何 $n$ 个线性无关的 $n$ 维向量都是 $\R^n$ 的最大无关组  

### 定理6 矩阵的秩等于它的列向量组的秩，也等于它的行向量组的秩
如果矩阵 $\boldsymbol{A}_{m \times n}$ 与 $\boldsymbol{B}_{l \times n}$ 的行向量组等价(通过行变换 $A \overset{r}{\sim} B$ )，这时方程 $\boldsymbol{Ax} = \boldsymbol{0}$ 与 $\boldsymbol{Bx} = \boldsymbol{0}$ 同解，所以一般将 $\boldsymbol{A}$ 化为最简形，可以很方便的看出列向量间的线性关系  
那么为什么$\boldsymbol{Ax} = \boldsymbol{0}$ 与 $\boldsymbol{Bx} = \boldsymbol{0}$ 同解就能够看出线性关系呢，因为可以从代数的角度看待这个问题，实际上 $\boldsymbol{A},\boldsymbol{B}$ 表示同一个方程组，所以同解，如果你把 $x$ 看作 $A$ 的各个列向量间的线性组合的参数，那么显然经过线性变换的矩阵不改变列向量之间的线性关系，也就是  
$$
\begin{pmatrix}
\boldsymbol{a}_1 & \boldsymbol{a}_2 & \cdots & \boldsymbol{a}_m
\end{pmatrix}
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_m
\end{pmatrix} = 
\begin{pmatrix}
x_1\boldsymbol{a}_{1} & \cdots & x_m\boldsymbol{a}_{m} 
\end{pmatrix} = \boldsymbol{0}
$$
$$
\begin{pmatrix}
\boldsymbol{b}_1 & \boldsymbol{b}_2 & \cdots & \boldsymbol{b}_m
\end{pmatrix}
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_m
\end{pmatrix} = 
\begin{pmatrix}
x_1\boldsymbol{b}_{1} & \cdots & x_m\boldsymbol{b}_{m} 
\end{pmatrix} = \boldsymbol{0}
$$
同解，$x$ 不变，所以线性关系不变  
当然也有更好理解的角度，假设矩阵 $\boldsymbol{A}$ 中，有列向量间的线性关系为
$$
\boldsymbol{a}_i = \alpha_1\boldsymbol{a}_{r1}+\cdots+\alpha_r\boldsymbol{a}_{rr}
$$  
当矩阵做初等行变换使得 $\boldsymbol{A} \sim \boldsymbol{B}$ 时，即存在可逆矩阵 $\boldsymbol{P}$ 使得 $\boldsymbol{PA} = \boldsymbol{B}$，对于每个列向量而言  
$$
\boldsymbol{Pa}_i = \boldsymbol{P}(\alpha_1\boldsymbol{a}_{r1}+\cdots+\alpha_r\boldsymbol{a}_{rr})
$$
$$
\boldsymbol{Pa}_i = \alpha_1\boldsymbol{P}\boldsymbol{a}_{r1}+\cdots+\alpha_r\boldsymbol{P}\boldsymbol{a}_{rr}
$$
显然线性关系不变，个人认为第二种证明方法更加直观  
## 4、线性方程组的解的结构
在上一章中，已经学习过用矩阵的初等变换解线性方程组，以及为什么可以这样解，为什么初等变换不改变结果。当然还引入了秩的重要概念，通过秩，我们可以很好的判定矩阵的维度，进而判定线性方程组解的情况，由此建立了两个重要的定理
* $n$ 个未知数的齐次线性方程组 $\boldsymbol{Ax} = \boldsymbol{0}$ 有非零解的充分必要条件时系数矩阵的秩 $R(\boldsymbol{A}) < n$
* $n$ 个未知数的非齐次线性方程组 $\boldsymbol{Ax} = \boldsymbol{b}$ 有解的充分必要条件是 $R(\boldsymbol{A}) = R((\boldsymbol{A,b}))$，当 $R(\boldsymbol{A}) = R((\boldsymbol{A,b})) = n$ 时有惟一解，当 $R(\boldsymbol{A}) = R((\boldsymbol{A,b})) = r < n$ 时有无限解  

实际上第一个定理，有非零解意味着线性相关，所以 $R(\boldsymbol{A}) < n$  
第二个定理，当 $R(\boldsymbol{A}) < R((\boldsymbol{A,b}))$ 时说明  $R(\boldsymbol{A}) < R(\boldsymbol{b})$ ，低维无法表示高维；当 $R(\boldsymbol{A}) = R((\boldsymbol{A,b}))$ 时，说明 $R(\boldsymbol{A}) \ge R(\boldsymbol{b})$ 高维可以表示低维；惟一解说明解空间 ( $x$ 的空间) 每个维度都有确切的数值；无限解说明解空间的有些维度没能用上，对于不参与组成 $\boldsymbol{b}$ 的维度当然可以取任意值，也就是产生了自由变量，所以 $x$ 变成了无限解  
总而言之这两个定理还是非常容易解释的

---  
下面来讨论线性方程组的解  
先讨论齐次线性方程组的解  
设有齐次线性方程组  
$$
\boldsymbol{Ax} = \boldsymbol{0}
$$
解为
$$
\boldsymbol{x} = \boldsymbol{\xi} = (\xi_1, \xi_2, \cdots, \xi_n)^T
$$
$\boldsymbol{x}$ 为齐次线性方程组的解向量
### 性质1 若 $\boldsymbol{x} = \boldsymbol{\xi}_1, \boldsymbol{x} = \boldsymbol{\xi}_2$ 为齐次方程组的解，则 $\boldsymbol{x} = \boldsymbol{\xi}_1+\boldsymbol{\xi}_2$ 也是齐次方程组的解
$$
\boldsymbol{A}(\boldsymbol{\xi_1+\xi_2}) = \boldsymbol{A\xi_1}+\boldsymbol{A\xi_2} = \boldsymbol{0}
$$
### 性质2 若 $\boldsymbol{x} = \boldsymbol{\xi}$ 为齐次方程组的解，设 $k \in \R$，则 $\boldsymbol{x} = k\boldsymbol{\xi}$ 也是齐次方程组的解
$$
\boldsymbol{A}(k\boldsymbol{\xi}) = k(\boldsymbol{A\xi}) = k\boldsymbol{0} = \boldsymbol{0}
$$
### 基础解系  
结合性质1和性质2，如果将齐次方程组的所有解的集合记作 $S$，如果能够求得 $S$ 的最大线性无关组 $S_0: \boldsymbol{\xi_1},\boldsymbol{\xi_2},\cdots,\boldsymbol{\xi_l}$，那么显然齐次线性方程组 $\boldsymbol{Ax} = \boldsymbol{0}$ 的通解为  
$$\begin{gather*}
\boldsymbol{x} = k_1\boldsymbol{\xi_1}+k_2\boldsymbol{\xi_2}+\cdots+k_l\boldsymbol{\xi_l} & (k_i \in \R)
\end{gather*}
$$
而齐次线性方程组的解集 $S$ 的最大线性无关组 $S_0$ 称为齐次线性方程组的**基础解系**  
求基础解系，设有齐次线性方程组
$$
\boldsymbol{Ax} = \boldsymbol{0}
$$
设参数矩阵 $\boldsymbol{A}$ 的秩为 $r$，对 $\boldsymbol{A}$ 进行初等行变换得到行最简形 $\boldsymbol{B}$
$$
\boldsymbol{A} \overset{r}{\sim} \boldsymbol{B} = 
\begin{pmatrix}
1 & \cdots & 0 & b_{11} & \cdots & b_{1n-r} \\
\vdots&    & \vdots & \vdots  &   & \vdots  \\
0 & \cdots & 1 & b_{r1} & \cdots & b_{rn-r} \\
0 &        &   & \cdots &        & 0        \\
\vdots&    &   &        &        & \vdots   \\
0 &        &   & \cdots &        & 0        \\
\end{pmatrix}
$$
则有
$$
\begin{pmatrix}
x_1 \\
\vdots \\
x_r \\
x_{r+1} \\
x_{r+2} \\
\vdots \\
x_n
\end{pmatrix} = 
c_1
\begin{pmatrix}
-b_{11} \\
\vdots \\
-b_{r1} \\
1 \\
0 \\
\vdots \\
0
\end{pmatrix}+
c_2
\begin{pmatrix}
-b_{12} \\
\vdots \\
-b_{r2} \\
0 \\
1 \\
\vdots \\
0
\end{pmatrix}+
\cdots +
c_{n-r}
\begin{pmatrix}
-b_{1n-r} \\
\vdots \\
-b_{rn-r} \\
0 \\
0 \\
\vdots \\
1
\end{pmatrix}
$$
即
$$
\boldsymbol{x} = c_1\boldsymbol{\xi}_1+c_2\boldsymbol{\xi}_2+\cdots+c_{n-r}\boldsymbol{\xi}_{n-r}
$$
显然，$\boldsymbol{\xi}_1,\boldsymbol{\xi}_2,\cdots,\boldsymbol{\xi}_{n-r}$是方程组的基础解系  
### 定理7 设 $m \times n$ 矩阵 $\boldsymbol{A}$ 的秩 $R(\boldsymbol{A}) = r$，则 $n$ 元齐次线性方程组 $\boldsymbol{Ax} = \boldsymbol{0}$ 的解集 $S$ 的秩 $R_s = n-r$

上面讨论了齐次线性方程组的通解和基础解系，下面讨论非齐次线性方程组  
设有非齐次线性方程组
$$
\boldsymbol{Ax} = \boldsymbol{b}
$$
有以下性质  
### 性质3 设 $\boldsymbol{x} = \boldsymbol{\eta}_1, \boldsymbol{x} = \boldsymbol{\eta}_2$ 都是非齐次线性方程组(向量方程)的解，则 $\boldsymbol{x} = \boldsymbol{\eta}_1-\boldsymbol{\eta}_2$ 为对应齐次线性方程组 $\boldsymbol{Ax} = \boldsymbol{0}$ 的解
$$
\boldsymbol{A}(\boldsymbol{\eta}_1-\boldsymbol{\eta}_2) = \boldsymbol{A}\boldsymbol{\eta}_1-\boldsymbol{A}\boldsymbol{\eta}_2 = \boldsymbol{b}-\boldsymbol{b} = \boldsymbol{0}
$$
### 性质4 设 $\boldsymbol{x} = \boldsymbol{\eta}$ 是方程 $\boldsymbol{Ax} = \boldsymbol{b}$ 的解，设 $\boldsymbol{x} = \boldsymbol{\xi}$ 是方程 $\boldsymbol{Ax} = \boldsymbol{0}$ 的解，则方程 $\boldsymbol{x} = \boldsymbol{\xi}+\boldsymbol{\eta}$ 是方程 $\boldsymbol{Ax} = \boldsymbol{b}$ 的解
$$
\boldsymbol{A}(\boldsymbol{\xi}+\boldsymbol{\eta}) = \boldsymbol{A}\boldsymbol{\xi}+\boldsymbol{A}\boldsymbol{\eta} = \boldsymbol{0}+\boldsymbol{b} = \boldsymbol{b}
$$
也就是说，如果求得方程 $\boldsymbol{Ax} = \boldsymbol{b}$ 的一个解 $\eta^*$ (称为特解)，那么方程 $\boldsymbol{Ax} = \boldsymbol{b}$ 的通解为  
$$
\boldsymbol{x} = k_1\boldsymbol{\xi}_1+k_2\boldsymbol{\xi}_2+\cdots+k_{n-r}\boldsymbol{\xi}_{n-r}+\eta^*
$$
其中 $\boldsymbol{\xi}_1, \boldsymbol{\xi}_2, \cdots, \boldsymbol{\xi}_{n-r}$ 是方程 $\boldsymbol{Ax} = \boldsymbol{0}$ 的基础解系  
## 5、向量空间
### 定义6 设 $V$ 为 $n$ 维向量的集合，如果集合 $V$ 非空，且集合 $V$ 对于向量的加法及数乘两种运算封闭，就称集合 $V$ 为向量空间
$$
\boldsymbol{a} \in V,\boldsymbol{b} \in V, \lambda \in \R
$$
$$
\boldsymbol{a}+\boldsymbol{b} \in V, \lambda\boldsymbol{a} \in V
$$

特殊的，$n$ 元齐次线性方程组的解集
$$
S = \{\boldsymbol{x} | \boldsymbol{Ax}=\boldsymbol{0} \}
$$
是一个向量空间(称为齐次线性方程组的解空间)
特殊的，$n$ 元非齐次线性方程组的解集
$$
S = \{\boldsymbol{x} | \boldsymbol{Ax}=\boldsymbol{b} \}
$$
不是向量空间
一般的，由向量组 $\boldsymbol{a}_1, \boldsymbol{a}_2, \cdots, \boldsymbol{a}_n$产生的向量空间为
$$
L = \{\boldsymbol{x}|\lambda_1\boldsymbol{a}_1+\lambda_2\boldsymbol{a}_2+\cdots+\lambda_n\boldsymbol{a}_n,\lambda_i \in \R \}
$$
### 定义7 设有向量空间 $V_1$ 及 $V_2$ ，若 $V_1 \subseteq V_2$，称 $V_1$ 是 $V_2$的一个子空间
### 定义8 设 $V$ 为向量空间，如果 $r$ 个向量 $\boldsymbol{a}_1,\boldsymbol{a}_2,\cdots,\boldsymbol{a}_r \in V$，且满足
* $\boldsymbol{a}_1,\boldsymbol{a}_2,\cdots,\boldsymbol{a}_r$ 线性无关
* $V$ 中任一向量都可由 $\boldsymbol{a}_1,\boldsymbol{a}_2,\cdots,\boldsymbol{a}_r$ 线性表示
### 则，向量组 $\boldsymbol{a}_1,\boldsymbol{a}_2,\cdots,\boldsymbol{a}_r$ 称为向量空间 $V$ 的一个基，$r$ 称为向量空间 $V$ 的维数，并称 $V$ 为 $r$ 维向量空间  
若把向量空间 $V$ 看作向量组，则由最大无关组的等价定义可知， $V$ 的基就是向量组的最大无关组， $V$ 的维数就是向量组的秩  
若向量组 $\boldsymbol{a}_1,\boldsymbol{a}_2,\cdots,\boldsymbol{a}_r$ 是向量空间 $V$ 的一个基，则 $V$ 可以表示为
$$
V = \{\boldsymbol{x}|\lambda_1\boldsymbol{a}_1+\lambda_2\boldsymbol{a}_2+\cdots+\lambda_r\boldsymbol{a}_r,\lambda_i \in \R \}
$$
### 定义9 如果在向量空间 $V$ 中取定一个基 $\boldsymbol{a}_1,\boldsymbol{a}_2,\cdots,\boldsymbol{a}_r$ ，那么 $V$ 中任意一个向量 $\boldsymbol{x}$ 可惟一地表示为
$$
\boldsymbol{x} = \lambda_1\boldsymbol{a}_1+\lambda_2\boldsymbol{a}_2+\cdots+\lambda_r\boldsymbol{a}_r
$$
数组 $\lambda_1, \lambda_2, \cdots, \lambda_r$ 称为向量 $x$ 在基 $\boldsymbol{a}_1,\boldsymbol{a}_2,\cdots,\boldsymbol{a}_r$ 中的坐标  
特别的，在 $n$ 维向量空间 $\R$ 中取单位坐标向量组 $\boldsymbol{e}_1,\boldsymbol{e}_2,\cdots,\boldsymbol{e}_r$ ，则可以用 $x$ 的各个分量表示坐标，$\boldsymbol{e}_1,\boldsymbol{e}_2,\cdots,\boldsymbol{e}_r$ 也叫做 $\R$ 中的自然基  
### 基变换公式
在 $\R^3$ 中取定一个旧基 $\boldsymbol{a}_1,\boldsymbol{a}_2,\boldsymbol{a}_3$，再取一个新基 $\boldsymbol{b}_1,\boldsymbol{b}_2,\boldsymbol{b}_3$ ，则  
$$
\boldsymbol{A} = (\boldsymbol{a}_1, \boldsymbol{a}_2, \boldsymbol{a}_3) = (\boldsymbol{e}_1, \boldsymbol{e}_2, \boldsymbol{e}_3)\boldsymbol{A}
$$
$$
(\boldsymbol{e}_1, \boldsymbol{e}_2, \boldsymbol{e}_3) = (\boldsymbol{a}_1, \boldsymbol{a}_2, \boldsymbol{a}_3)\boldsymbol{A}^{-1}
$$
$$
\boldsymbol{B} = (\boldsymbol{e}_1, \boldsymbol{e}_2, \boldsymbol{e}_3)\boldsymbol{B} = 
(\boldsymbol{a}_1, \boldsymbol{a}_2, \boldsymbol{a}_3)\boldsymbol{A}^{-1}\boldsymbol{B}
$$
则存在 $\boldsymbol{P} = \boldsymbol{A}^{-1}\boldsymbol{B}$ 使得
$$
(\boldsymbol{b}_1, \boldsymbol{b}_2, \boldsymbol{b}_3) = 
(\boldsymbol{a}_1, \boldsymbol{a}_2, \boldsymbol{a}_3)\boldsymbol{P}
$$
### 坐标变换公式
在 $\R^3$ 中取定一个旧基 $\boldsymbol{a}_1,\boldsymbol{a}_2,\boldsymbol{a}_3$，再取一个新基 $\boldsymbol{b}_1,\boldsymbol{b}_2,\boldsymbol{b}_3$ ，设向量 $x$ 在旧基中的坐标为 $(x_1, x_2, x_3)$，在新基中的坐标为 $(y_1, y_2, y_3)$，则有  
$$
(\boldsymbol{a}_1,\boldsymbol{a}_2,\boldsymbol{a}_3)
\begin{pmatrix}
x_1 \\
x_2 \\
x_3
\end{pmatrix} = 
(\boldsymbol{b}_1,\boldsymbol{b}_2,\boldsymbol{b}_3)
\begin{pmatrix}
y_1 \\
y_2 \\
y_3
\end{pmatrix}
$$
$$
\begin{pmatrix}
y_1 \\
y_2 \\
y_3
\end{pmatrix} = 
\boldsymbol{B}^{-1}\boldsymbol{A}
\begin{pmatrix}
x_1 \\
x_2 \\
x_3
\end{pmatrix}
$$
$$
\begin{pmatrix}
y_1 \\
y_2 \\
y_3
\end{pmatrix} = 
\boldsymbol{P}^{-1}
\begin{pmatrix}
x_1 \\
x_2 \\
x_3
\end{pmatrix}
$$
# 第5章 相似矩阵及二次型
## 1、向量的内积、长度、正交性
### 定义1 内积  
设有 $n$ 维向量
$$
\boldsymbol{x} = (x_1, x_2, \cdots, x_n)^T,
\boldsymbol{y} = (y_1, y_2, \cdots, y_n)^T,
$$
令
$$
\begin{bmatrix} \boldsymbol{x}, \boldsymbol{y} \end{bmatrix} = x_1y_1+x_2y_2+\cdots+x_ny_n
$$
$[\boldsymbol{x}, \boldsymbol{y}]$ 称为 $\boldsymbol{x}$ 与 $\boldsymbol{y}$ 的内积  
内积是两个向量之间的一种运算，其结果是一个实数，当 $\boldsymbol{x}$ 与 $\boldsymbol{y}$ 都为列向量时，有
$$
\begin{bmatrix} \boldsymbol{x}, \boldsymbol{y} \end{bmatrix} = \boldsymbol{x}^T\boldsymbol{y}
$$
### 内积具有以下性质
* $[\boldsymbol{x}, \boldsymbol{y}] = [\boldsymbol{y}, \boldsymbol{x}]$
* $[\lambda\boldsymbol{x}, \boldsymbol{y}] = \lambda[\boldsymbol{x}, \boldsymbol{y}]$
* $[\boldsymbol{x+y}, \boldsymbol{z}] = [\boldsymbol{x}, \boldsymbol{z}]+[\boldsymbol{y}, \boldsymbol{z}]$
* 当 $\boldsymbol{x}=0$ 时，$[\boldsymbol{x}, \boldsymbol{x}]=0$ ，当 $\boldsymbol{x} \ne 0$ 时，$[\boldsymbol{x}, \boldsymbol{x}] > 0$  
### 施瓦茨(Schwarz)不等式
$$
[\boldsymbol{x}, \boldsymbol{y}]^2 \le [\boldsymbol{x}, \boldsymbol{x}][\boldsymbol{y}, \boldsymbol{y}]
$$
### 解析几何，向量的数量积
$$
\boldsymbol{x} \cdot \boldsymbol{y} = |\boldsymbol{x}||\boldsymbol{y}| \cos{\theta}
$$
在直角坐标系中有
$$
\boldsymbol{x} \cdot \boldsymbol{y} = (x_1, x_2, \cdots, x_n) \cdot (y_1, y_2, \cdots, y_n) = 
x_1y_1+x_2y_2+\cdots+x_ny_n
$$
$n$ 维向量的内积是数量积的一种推广，但是3维以上向量的长度和夹角并不直观，因此只能按照数量积的坐标公式计算来推广，反过来再利用内积来定义长度和夹角  
这里还需要强调，不要忘记向量数量积 $(a \cdot b)$ 的几何意义是 $a$ 向量在 $b$ 向量方向上的投影乘 $b$ 向量的模长(向量的长度)
### 定义2 向量的长度
$$
\begin{Vmatrix}
\boldsymbol{x}
\end{Vmatrix} = 
\sqrt{[\boldsymbol{x},\boldsymbol{x}]} = 
\sqrt{x_1^2+x_2^2+\cdots+x_n^2}
$$
$\begin{Vmatrix} \boldsymbol{x} \end{Vmatrix}$ 称作 $n$ 维向量 $x$ 的范数(长度)  
向量的长度具有以下性质
* 非负性，当 $\boldsymbol{x} \ne 0$ 时，$\begin{Vmatrix} \boldsymbol{x} \end{Vmatrix} > 0$，当 $\boldsymbol{x} = 0$ 时，$\begin{Vmatrix} \boldsymbol{x} \end{Vmatrix} = 0$  
* 齐次性 $\begin{Vmatrix} \lambda\boldsymbol{x} \end{Vmatrix} = |\lambda|\begin{Vmatrix} \boldsymbol{x} \end{Vmatrix}$  

当 $\begin{Vmatrix} \boldsymbol{x} \end{Vmatrix} = 1$ 时，称 $\boldsymbol{x}$ 为单位向量，若 $\boldsymbol{a} \ne \boldsymbol{0}$，取单位向量
$$\boldsymbol{x} = \frac{\boldsymbol{a}}{\begin{Vmatrix} \boldsymbol{a} \end{Vmatrix}}$$
此过程称为把向量 $\boldsymbol{a}$ 单位化  
由施瓦茨不等式，有
$$
-1 \le \frac{[\boldsymbol{x},\boldsymbol{y}]}{\begin{Vmatrix} \boldsymbol{x} \end{Vmatrix}\begin{Vmatrix} \boldsymbol{y} \end{Vmatrix}} \le 1,
\begin{Vmatrix} \boldsymbol{x} \end{Vmatrix}\begin{Vmatrix} \boldsymbol{y} \end{Vmatrix} \ne 0
$$
于是有向量夹角的定义，当 $\boldsymbol{x} \ne 0, \boldsymbol{y} \ne 0$ 时  
$$
\cos\theta = \frac{[\boldsymbol{x},\boldsymbol{y}]}{\begin{Vmatrix} \boldsymbol{x} \end{Vmatrix}\begin{Vmatrix} \boldsymbol{y} \end{Vmatrix}}
$$
$$
\theta = \arccos\frac{[\boldsymbol{x},\boldsymbol{y}]}{\begin{Vmatrix} \boldsymbol{x} \end{Vmatrix}\begin{Vmatrix} \boldsymbol{y} \end{Vmatrix}}
$$  
$\theta$ 称为 $n$ 维向量 $x$ 与 $y$ 的夹角  
当 $[\boldsymbol{x},\boldsymbol{y}] = 0$ 时，称向量 $\boldsymbol{x}$ 与 $\boldsymbol{y}$ 正交，显然，若 $\boldsymbol{x=0}$，则 $\boldsymbol{x}$ 与任何向量都正交  
正交向量组：一组两两正交的非零向量  
### 正交向量组的性质
### 定理1 若 $n$ 维向量 $\boldsymbol{a}_1,\boldsymbol{a}_2,\cdots,\boldsymbol{a}_r$ 是一组两两正交的非零向量，则 $\boldsymbol{a}_1,\boldsymbol{a}_2,\cdots,\boldsymbol{a}_r$ 线性无关  
设有 $\lambda_1, \lambda_2, \cdots, \lambda_r$ 使得
$$
\lambda_1\boldsymbol{a}_1+\lambda_2\boldsymbol{a}_2+\cdots+\lambda_r\boldsymbol{a}_r = 0
$$
两边同时与 $\boldsymbol{a}_1$ 做内积  
$$
\lambda_1[\boldsymbol{a}_1, \boldsymbol{a}_1] = 0
$$
由于 $\boldsymbol{a}_1$ 是非零向量，所以 $\lambda_1 = 0$  
同理可得 $\lambda_1 = \lambda_2 = \cdots = \lambda_r = 0$  
即，$\boldsymbol{a}_1,\boldsymbol{a}_2,\cdots,\boldsymbol{a}_r$ 线性无关  
其实不难想象，如果一组向量两两正交，那么必然每个向量都有自己"独有"的维度，而自己的维度是不能被其他向量所表示的，所以正交向量组一定是线性无关  
### 定义3 设 $n$ 维向量 $\boldsymbol{e}_1,\boldsymbol{e}_2,\cdots,\boldsymbol{e}_r$ 是向量空间 $V,V \subseteq \R^n$ 的一个基，如果 $\boldsymbol{e}_1,\boldsymbol{e}_2,\cdots,\boldsymbol{e}_r$ 两两正交，且都是单位向量 (模长=1)，则称 $\boldsymbol{e}_1,\boldsymbol{e}_2,\cdots,\boldsymbol{e}_r$ 是 $V$ 的一个标准正交基
若 $\boldsymbol{e}_1,\boldsymbol{e}_2,\cdots,\boldsymbol{e}_r$ 是 $V$ 的一个标准正交基，那么 $V$ 中任一向量 $\boldsymbol{a}$ 应能由 $\boldsymbol{e}_1,\boldsymbol{e}_2,\cdots,\boldsymbol{e}_r$ 线性表示，即  
$$
\boldsymbol{a} = \lambda_1\boldsymbol{e}_1+\lambda_2\boldsymbol{e}_2+\cdots+\lambda_r\boldsymbol{e}_r
$$
为了求出系数 $\lambda_i$，我们两边同时乘 $\boldsymbol{e}_i^T$  
$$
\boldsymbol{e}_i^T\boldsymbol{a} = \lambda_i\boldsymbol{e}_i^T\boldsymbol{e}_i = \lambda_i
$$
$$
\lambda_i = \boldsymbol{e}_i^T\boldsymbol{a} = [\boldsymbol{a},\boldsymbol{e}_i]
$$
通过这个公式，可以方便的求解向量在标准正交基中的**坐标**  

### 施密特(Schmidt)正交化  
设 $\boldsymbol{a}_1, \cdots, \boldsymbol{a}_r$ 是向量空间 $V$ 中的一个基，要求 $V$ 的一个标准正交基，也就是找一组两两正交的单位向量 $\boldsymbol{e}_1,\cdots,\boldsymbol{e}_r$，使得 $\boldsymbol{e}_1,\cdots,\boldsymbol{e}_r$ 与 $\boldsymbol{a}_1, \cdots, \boldsymbol{a}_r$ 等价，这个过程称为，把基 $\boldsymbol{a}_1, \cdots, \boldsymbol{a}_r$ 标准正交化  
将 $\boldsymbol{a}_1, \cdots, \boldsymbol{a}_r$ 标准正交化，我们可以取  
$$
\boldsymbol{b}_1 = \boldsymbol{a}_1
$$
$$
\boldsymbol{b}_2 = \boldsymbol{a}_2-\frac{[\boldsymbol{b}_1, \boldsymbol{a}_2]}{[\boldsymbol{b}_1,\boldsymbol{b}_1]}\boldsymbol{b}_1
$$
$$
\cdots \ \cdots \ \cdots
$$
$$
\boldsymbol{b}_r  = \boldsymbol{a}_r-
\frac{[\boldsymbol{b}_1, \boldsymbol{a}_r]}{[\boldsymbol{b}_1,\boldsymbol{b}_1]}\boldsymbol{b}_1-
\frac{[\boldsymbol{b}_2, \boldsymbol{a}_r]}{[\boldsymbol{b}_2,\boldsymbol{b}_2]}\boldsymbol{b}_2-
\cdots -
\frac{[\boldsymbol{b}_{r-1}, \boldsymbol{a}_r]}{[\boldsymbol{b}_{r-1},\boldsymbol{b}_{r-1}]}\boldsymbol{b}_{r-1}
$$
$$
\boldsymbol{b}_k = \boldsymbol{a}_k-\sum_{i=1}^{k-1}\frac{[\boldsymbol{b}_i, \boldsymbol{a}_k]}{[\boldsymbol{b}_i,\boldsymbol{b}_i]}\boldsymbol{b}_i
$$  
容易验证 $\boldsymbol{b}_1,\cdots,\boldsymbol{b}_r$ 两两正交，且 $\boldsymbol{b}_1,\cdots,\boldsymbol{b}_r$ 与 $\boldsymbol{a}_1, \cdots, \boldsymbol{a}_r$ 等价  
然后将 $\boldsymbol{b}_1,\cdots,\boldsymbol{b}_r$ 单位化，即取  
$$
\boldsymbol{e}_1 = \frac{1}{\begin{Vmatrix} \boldsymbol{b}_1 \end{Vmatrix}}\boldsymbol{b}_1,
\boldsymbol{e}_2 = \frac{1}{\begin{Vmatrix} \boldsymbol{b}_2 \end{Vmatrix}}\boldsymbol{b}_2,
\cdots,
\boldsymbol{e}_r = \frac{1}{\begin{Vmatrix} \boldsymbol{b}_r \end{Vmatrix}}\boldsymbol{b}_r,
$$
求得的 $\boldsymbol{e}_1,\cdots,\boldsymbol{e}_r$ 就是 $V$ 的一个标准正交基  
上述从线性无关向量组  $\boldsymbol{a}_1, \cdots, \boldsymbol{a}_r$ 导出正交向量组  $\boldsymbol{b}_1, \cdots, \boldsymbol{b}_r$ 的过程称为施密特(Schmidt)正交化  
它不仅满足 $\boldsymbol{b}_1,\cdots,\boldsymbol{b}_r$ 与  $\boldsymbol{a}_1, \cdots, \boldsymbol{a}_r$ 等价，还满足对任何 $k(1 \le k \le r)$，向量组 $\boldsymbol{b}_1,\cdots,\boldsymbol{b}_k$ 与  $\boldsymbol{a}_1, \cdots, \boldsymbol{a}_k$ 等价  
### 定义4 如果 $n$ 阶矩阵 $\boldsymbol{A}$ 满足  
$$
\boldsymbol{A}^T\boldsymbol{A} = \boldsymbol{I} , (\boldsymbol{A}^T = \boldsymbol{A}^{-1})
$$  
### 那么称 $\boldsymbol{A}$ 为正交矩阵，简称正交阵  
不难看出，当  
$$
\begin{pmatrix}
\boldsymbol{a}_1^T \\
\boldsymbol{a}_2^T \\
\vdots \\
\boldsymbol{a}_n^T
\end{pmatrix}
\begin{pmatrix}
\boldsymbol{a}_1
\boldsymbol{a}_2
\cdots 
\boldsymbol{a}_n
\end{pmatrix} = 
\boldsymbol{I}
$$  
即
$$
\boldsymbol{a}_i^T\boldsymbol{a}_j = 
\begin{cases}
1 & i=j \\
0 & i \ne j
\end{cases}
$$  
方阵 $\boldsymbol{A}$ 为正交矩阵的充分必要条件是 $\boldsymbol{A}$ 的列向量都是单位向量，且两两正交  
因为 $\boldsymbol{A}^T\boldsymbol{A} = \boldsymbol{I}$ 与 $\boldsymbol{A}\boldsymbol{A}^T = \boldsymbol{I}$ 等价，所以上述结论对 $\boldsymbol{A}$ 的行向量依然成立  
由此可见，$n$ 阶正交矩阵 $\boldsymbol{A}$ 的 $n$ 个列(行)向量构成向量空间 $\R^n$ 的一个标准正交基  
### 定义5 若 $\boldsymbol{P}$ 为正交矩阵，则线性变换 $\boldsymbol{y} = \boldsymbol{Px}$ 称为正交变换
$$
\begin{Vmatrix}
\boldsymbol{y}
\end{Vmatrix} = 
\sqrt{\boldsymbol{y}^T\boldsymbol{y}} = 
\sqrt{\boldsymbol{x}^T\boldsymbol{P}^T\boldsymbol{P}\boldsymbol{x}} = 
\sqrt{\boldsymbol{x}^T\boldsymbol{x}} = 
\begin{Vmatrix}
\boldsymbol{x}
\end{Vmatrix}
$$
由于 $\begin{Vmatrix} \boldsymbol{y} \end{Vmatrix}$ 表示向量长度，显然在正交变换中，线段的长度保持不变  
## 2、方阵的特征值与特征向量
### 定义6 设 $\boldsymbol{A}$ 是 $n$ 阶矩阵，如果数 $\lambda$ 和 $n$ 维非零列向量 $x$ 使下面的关系式成立
$$
\boldsymbol{Ax} = \lambda\boldsymbol{x}
$$  
### 数 $\lambda$ 称为矩阵 $\boldsymbol{A}$ 的特征值，非零向量 $\boldsymbol{x}$ 称为 $\boldsymbol{A}$ 的对应于特征值 $\lambda$ 的特征向量
上式也可以写成
$$
(\boldsymbol{A}-\lambda\boldsymbol{I})\boldsymbol{x} = \boldsymbol{0}
$$
这是一个齐次线性方程组，它有非零解的从充分必要条件是系数行列式等于 $0$，即使矩阵 $\boldsymbol{A}-\lambda\boldsymbol{I}$ 的所有列向量线性相关  
$$
\begin{vmatrix}
\boldsymbol{A}-\lambda\boldsymbol{I}
\end{vmatrix} = 
0
$$
即
$$
\begin{vmatrix}
a_{11}-\lambda & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22}-\lambda & \cdots & a_{2n} \\
\vdots & \vdots & & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nn}-\lambda \\
\end{vmatrix} = 0
$$
这个行列式$=0$的式子称为矩阵 $\boldsymbol{A}$ 的特征方程，其左端的 $\begin{vmatrix} \boldsymbol{A}-\lambda\boldsymbol{I} \end{vmatrix}$ 是 $\lambda$ 的 $n$ 次多项式，记作 $f(\lambda)$，称为矩阵 $\boldsymbol{A}$ 的特征多项式  
显然，$\boldsymbol{A}$ 的特征值就是特征方程的解，特征方程在复数范围内恒有解，其个数为方程的次数(解方程)，因此，$n$ 阶矩阵 $A$ 在复数范围内有 $n$ 个特征值  
设 $n$ 阶矩阵 $\boldsymbol{A}=(a_{ij})$ 的特征值为 $\lambda_1,\lambda_2,\cdots, \lambda_n$，不难证明：  
* $\lambda_1+\lambda_2+\cdots+\lambda_n = a_{11}+a_{22}+\cdots+a_{nn}$
* $\lambda_1\lambda_2\cdots\lambda_n = \begin{vmatrix} \boldsymbol{A} \end{vmatrix} $  

证明：
因为
$$
\begin{vmatrix}
\boldsymbol{A}-\lambda\boldsymbol{I}
\end{vmatrix} = 
\boldsymbol{0}
$$
不妨令
$$
\begin{vmatrix}
\lambda\boldsymbol{I}-\boldsymbol{A}
\end{vmatrix} = \boldsymbol{0}
$$
$$
\begin{vmatrix}
\lambda\boldsymbol{I}-\boldsymbol{A}
\end{vmatrix} = 
(\lambda-\lambda_1)(\lambda-\lambda_2)\cdots(\lambda-\lambda_n)
$$
将式子展开
$$
\begin{vmatrix}
\lambda\boldsymbol{I}-\boldsymbol{A}
\end{vmatrix} = 
\lambda^n-(\lambda_1+\lambda_2+\cdots+\lambda_n)\lambda^{n-1}+\cdots+(-1)^n(\lambda_1\lambda_2\cdots\lambda_n)
$$
对行列式不断地用第一行的第一个元素展开，不难发现
$$
\begin{vmatrix}
\lambda\boldsymbol{I}-\boldsymbol{A}
\end{vmatrix} = 
(\lambda-a_{11})(\lambda-a_{22})\cdots(\lambda-a_{nn})+(\lambda-a_{11})(\lambda-a_{22})\cdots(-a_{n-1n}a_{nn-1})+\cdots
$$
$$
\begin{vmatrix}
\lambda\boldsymbol{I}-\boldsymbol{A}
\end{vmatrix} = 
(\lambda-a_{11})(\lambda-a_{22})\cdots(\lambda-a_{nn})+\cdots
$$
只有第一项 $(\lambda-a_{11})(\lambda-a_{22})\cdots(\lambda-a_{nn})$ 包含 $n$ 次和 $n-1$次  
对比系数法  
对 $\lambda^n$ 的系数而言
$$
1 = 1
$$
所以有
$$
-\sum_{i=1}^n\lambda_i = -\sum_{i=1}^na_{ii}
$$
$$
\lambda_1+\lambda_2+\cdots+\lambda_n = a_{11}+a_{22}+\cdots+a_{nn}
$$  
令 $\lambda = 0$，则有  
$$
\begin{vmatrix}
\lambda\boldsymbol{I}-\boldsymbol{A}
\end{vmatrix} = 
(-1)^n
\begin{vmatrix}
\boldsymbol{A}
\end{vmatrix} = 
(-1)^n(\lambda_1\lambda_2\cdots\lambda_n)
$$
$$
\lambda_1\lambda_2\cdots\lambda_n = \begin{vmatrix} \boldsymbol{A} \end{vmatrix} 
$$
这里可以知道矩阵 $\boldsymbol{A}$ 可逆的充分必要条件是 $n$ 个特征值全不为 $0$  
设 $\lambda = \lambda_i$ 为矩阵 $\boldsymbol{A}$ 的一个特征值，则由方程  
$$
(\boldsymbol{A}-\lambda_i\boldsymbol{I})\boldsymbol{x} = \boldsymbol{0}
$$
可以求得非零解 $\boldsymbol{x=p_i}$，那么 $\boldsymbol{p_i}$ 是 $\boldsymbol{A}$ 的对应于特征值 $\lambda_i$ 的特征向量  
若 $\boldsymbol{p_i}$ 是矩阵的特征向量，则 $k\boldsymbol{p_i}$ 也是矩阵的特征向量  
* 若 $\lambda$ 是 $\boldsymbol{A}$ 的特征值，则 $\lambda^k$ 是 $\boldsymbol{A}^k$ 的特征值
* 若 $\lambda$ 是 $\boldsymbol{A}$ 的特征值，则 $\varphi(\lambda)$ 是 $\varphi(\boldsymbol{A})$ 的特征值  
### 定理2 设 $\lambda_1,\lambda_2,\cdots,\lambda_m$ 是方阵 $\boldsymbol{A}$ 的 $m$ 个特征值，$\boldsymbol{p_1},\boldsymbol{p_2},\cdots,\boldsymbol{p_n}$ 依次是与之对应的特征向量，如果 $\lambda_1,\lambda_2,\cdots,\lambda_m$ 各不相等，则 $\boldsymbol{p_1},\boldsymbol{p_2},\cdots,\boldsymbol{p_n}$ 线性无关  
### 推论 设 $\lambda_1,\lambda_2$ 分别是方阵 $\boldsymbol{A}$ 的两个不同的特征值，$\xi_1,\xi_2,\cdots,\xi_s$ 和 $\eta_1,\eta_2,\cdots,\eta_r$ 分别是对应于 $\lambda_1$ 和 $\lambda_2$ 的线性无关的特征向量，则 $\xi_1,\xi_2,\cdots,\xi_s,\eta_1,\eta_2,\cdots,\eta_r$ 线性无关