#coding:utf-8
#设损失函数 loss=(w+1)^2，零w初值是常数5。反向传播就是求最优w，即求最小loss对应的w值
import tesorflow as tf
#定义待优化参数w初值为5
w = tf.Variable(tf.constant(5, dtype=tf.float32))
#定义损失函数loss
loss = tf.square(w+1)
#定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#生成会话，训练40轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    w_val = sess.run(w)
    loss_val = sess.run(loss)
    print "After %s step(s): w is %s, loss is %s" % (i, w_val, loss_val)
