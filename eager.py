import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
a = tf.constant(1, name = 'a')
b = tf.constant(2, name = 'b')

print(a)
result = a + b
print(result)

#with tf.Session() as sess:
#    print(sess.run(result))
