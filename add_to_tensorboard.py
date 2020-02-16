#https://github.com/tensorflow/tensorflow/issues/8854
import tensorflow as tf
from tensorflow.python.platform import gfile
path = "D:/Experiments/tensorflowQuantize/"
for model in ["he", "cao", "santara", "luo_cnn"]:
    with tf.Session() as sess:
        model_filename = path + model + '.pb'
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)
LOGDIR=path
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)