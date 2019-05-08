import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 2], stddev=1, seed=1))

#x = tf.constant([[0.7, 0.9]])
x = tf.placeholder(tf.float32, shape=(2,2), name='input')
b = tf.placeholder(tf.float32, shape=(1,2), name='input_1')
b1 = tf.placeholder(tf.float32, shape=(2,1), name='input_1')

a = tf.matmul(x, w1)
y = tf.matmul(a, w2) + b
y1 = tf.matmul(a, w2)
y2 = tf.matmul(a, w2) + b1

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('D:/work/pytest/mnist/log_folder', sess.graph )
    sess.run(w1.initializer)
    sess.run(w2.initializer)
    print(w1)
    print(w2)
    print(sess.run(y1, feed_dict={x:[[0.7, 0.9],[0.4, 0.5]],b:[[1, 2]]}))
    print(sess.run(y, feed_dict={x:[[0.7, 0.9],[0.4, 0.5]],b:[[1, 2]]}))
    print(sess.run(y2, feed_dict={x:[[0.7, 0.9],[0.4, 0.5]],b1:[[1], [2]]}))