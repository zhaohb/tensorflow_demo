
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python import debug as tf_debug


w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1), name='var_1')
print(w1)
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1), name='var_2')
print(w2)
#x = tf.constant([[0.7, 0.9]])
x = tf.placeholder(tf.float32, shape=(1,2), name='input')
print(x)
a = tf.matmul(x, w1, name='mat_a')
print(a)
y = tf.matmul(a, w2, name='mat_y')
print(y)

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('D:/work/pytest/mnist/log_folder', sess.graph )
    sess.run(w1.initializer)
    sess.run(w2.initializer)
    print(w1)
    print(w2)
    print(sess.run(y, feed_dict={x:[[0.7, 0.9]]}))
