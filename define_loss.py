import tensorflow as tf
from numpy.random import RandomState
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 8

#通过shape是(None, 2)的X来预测应该生产的数量
x = tf.placeholder(tf.float32, shape=(None, 2), name = 'x-input')
#y_ 真实的生产数量
y_ = tf.placeholder(tf.float32, shape=(None, 1), name = 'y-input')

w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
#y 预测的生产数量
y = tf.matmul(x, w1)

loss_less = 10
loss_more = 1

#自定义损失函数：y>y_时每个损失(y - y_) * 1， y<y_时损失(y_ - y) * 10
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_)*loss_more, (y_ - y)*loss_less))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[x1 + x2 + rdm.rand()/10.0 - 0.05] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 15000
    for i in range(STEPS):
        start = (i*batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i%1000 ==0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print(" After %d training step(s), loss on all data is %g" %(i, total_loss))
    print(sess.run(w1))
