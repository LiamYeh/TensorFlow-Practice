# TensorFlow-Practice

## 激活函数
> 激活函数的作用是引入非线性的因素，由于线性模型的表达能力不够，加入激活函数能解决线性模型所不能解决的问题
<br> 一般线性模型的函数为 y = Weight * x + biases
#### 常见的激活函数主要有以下这些：
- sigmoid 函数
- tanh函数
- relu 函数
- softplus 函数
- leakrelu 函数
- ELU 函数
- SELU函数
#### 具体内容参见以下链接
> (http://blog.csdn.net/u011630575/article/details/78063641)
> (https://www.zhihu.com/question/22334626/answer/21036590)
## 损失函数 代价函数 目标函数
> 每一个算法都有一个目标函数（objective function），算法就是让这个目标函数达到最优
<br> 损失函数体现好坏,损失函数越小，其越接近目标函数
<br> 代价函数与损失函数相同 不过代价函数表现的是总体的好坏
- 损失函数，一般是针对单个样本
> ![](https://www.zhihu.com/equation?tex=%5Cleft%7C+y_i-f%28x_i%29+%5Cright%7C)
- 代价函数, 一般是针对总体
> ![](https://www.zhihu.com/equation?tex=1%2FN.%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7B%5Cleft%7C+y_i-f%28x_1%29+%5Cright%7C%7D)
- 目标函数, 所期望的最优函数
> ![](https://www.zhihu.com/equation?tex=1%2FN.%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7B%5Cleft%7C+y_i-f%28x_1%29+%5Cright%7C%7D+%2B+%E6%AD%A3%E5%88%99%E5%8C%96%E9%A1%B9)
