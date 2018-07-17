#coding:utf-8
#预测多或预测少的影响一样
#0导入模块，生成数据集
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455
COST = 1
PROFIT = 9

rdm = np.random.RandomState(SEED)
X = rdm.rand(32,2)
Y_ = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1,x2) in X]

#1定义神经网络的输入、参数和输出，定义前向传播过程。
x = tf.placeholder(tf.float32, shape=(None,2))
y_ = tf.placeholder(tf.float32, shape=(None,1))
w1 = tf.Variable(tf.random_normal([2,1], stddev=1, seed=1))
y = tf.matmul(x, w1)

#2定义损失函数及反向传播方法。
#定义损失函数使得预测少了的损失大，于是模型应该偏向多的方向预测
loss = tf.reduce_sum(tf.where(tf.greater(y,y_), (y-y_)*COST,(y-y_)*PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#3.生成会话，训练STEP轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 200
    for i in range(STEPS):
        start = (i*BATCH_SIZE) %32
        end = (i*BATCH_SIZE) %32 + BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
        if i % 5 == 0:
            print("After %d train step(s), w1 is :" % (i))
            print(sess.run(w1), "\n")
    print("Final w1 is: \n", sess.run(w1))

