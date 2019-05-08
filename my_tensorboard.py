
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

input1 = tf.constant([1, 2, 3], dtype=tf.float32, name='input1')
input2 = tf.Variable(tf.random_uniform([3]), dtype=tf.float32, name='input2')
output = tf.add_n([input1, input2], name='add')
model  = tf.initialize_all_variables()

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('D:/work/pytest/mnist/log_folder', sess.graph )
    sess.run(model)
    print(sess.run(output))

server = tf.train.Server(...)
with tf.Session(server.target):
