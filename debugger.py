import tensorflow as tf
from tensorflow.python import debug as tf_debug
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.Session()
sess = tf_debug.TensorBoardDebugWrapperSession(sess, "zhaohongbo00:6007")
sess.run(my_fetches)