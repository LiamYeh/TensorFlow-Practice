
# coding: utf-8

# In[2]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[8]:

#生成200随机点
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis] #生成一个新维度
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise # y=x^2+随机值


# In[10]:

# tf.Variable：用于一些可训练变量（trainable variables），比如模型的权重（weights，W）或者偏执值（bias）
# 声明时，必须提供初始值
# tf.placeholder：用于得到传递进来的真实的训练样本，用作占位符
# 不必指定初始值，在运行时通过 Session.run 函数的 feed_dict 参数指定

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])


# In[14]:

#定义神经网络中间层
Weight_L1 = tf.Variable(tf.random_normal([1,10])) #中间层权重 中间层链接了一个输入神经元和十个中间神经元
biases_L1 = tf.Variable(tf.zeros([1,10])) #中间层偏置值
Wx_plus_b_L1 = tf.matmul(x,Weight_L1) + biases_L1 # y = Weight * x + biases
L1 = tf.nn.tanh(Wx_plus_b_L1) #中间层采用tanh激活函数


# In[16]:

#定义输出层
Weight_L2 = tf.Variable(tf.random_normal([10,1])) #输出层权重 输出层链接了十个中间神经元和一个输出神经元
biases_L2 = tf.Variable(tf.zeros([1,1])) #输出层偏置值
Wx_plus_b_L2 = tf.matmul(L1,Weight_L2) + biases_L2 #输出层的输入相当于中间层 即L1
prediction = tf.nn.tanh(Wx_plus_b_L2) #输出层也采用tanh激活函数


# In[19]:

#定义损失函数和梯度下降法
loss = tf.reduce_mean(tf.square(y-prediction)) # loss = (y - prediction)^2的平均值
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) #采用梯度下降法，下降速度为0.1，下降方向为最小化


# In[20]:

#进行训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #初始化所有变量
    for _ in range(2000): #训练2000次
        sess.run(train_step,feed_dict={x:x_data,y:y_data}) #开始训练 传入训练值x_data y_data
        
    #获得预测值
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    
    #画图
    plt.figure()
    plt.scatter(x_data,y_data) #散点图
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()

