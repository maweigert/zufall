import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from isonet.nets import resunet_model

fname = "models/resunet.hdf5"

K.set_learning_phase(0)

if not "model" in locals():
    model = resunet_model((None,None,1),
                         "relu",
                         2, 32, 9, 9, n_conv_per_depth=2)

    model.load_weights(fname)

output_layer = tf.identity(model.output, name="output")

sess = K.get_session()

graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ["output"])
graph_io.write_graph(graph, "tf_models","resunet.pb", as_text=False)
