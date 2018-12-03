#
# Genie
# Tensorflow Model file Generation
# Date : 08/05/2018
# Copyright: Ge3f Pte Ltd
#

# #### Export protobuf file
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from tensorflow.python.framework.graph_util import convert_variables_to_constants

model = load_model(
    "D:/Data_256x256/weights-improvement-04-0.9879.hdf5")
K.set_learning_phase(0)
[node.op.name for node in model.inputs]
[node.op.name for node in model.outputs]
session = K.get_session()
min_graph = convert_variables_to_constants(
    session, session.graph_def, [node.op.name for node in model.outputs])
data_dir = "C:/Users/Lenovo/Desktop/"
tf.train.write_graph(min_graph, data_dir, "BVe2_18_6.pb", as_text=False)
session.close()
