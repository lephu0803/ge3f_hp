#
# Genie
# Tensorflow Model Prediction
# Date : 08/05/2018
# Copyright: Ge3f Pte Ltd
#

import cv2
import numpy as np
import tensorflow as tf
import time
from datetime import timedelta
from PIL import Image
import glob
import scipy.misc
import sys  
import keras.backend as K

model_file = "/media/aimonsters/Seagate5TB/FCI/512_Augmentation/B1_Pos1/B1_Pos1.pb"
#path_test_images = "C:/Users/Lenovo/Desktop/TestSet18_6/05_GoodLens/*.bmp"
classes = ["01_Dust_SmallFM","02_Good","03_SlotChip"]

thresholds = [0.1, 0.1, 0.1]
[width, height] = [512, 512]
#mean = 94.216579534425989
scale = 0.00392157

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.85
K.tensorflow_backend.set_session(tf.Session(config=config))

# Load tensorflow model


def load_model():
    with tf.gfile.GFile(model_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph


def predict(session, X_test, Y_test, file_test_images, verbose=0):
    count = 0
    test_acc = 0
    time_diff = 0
    global_labels = []
    label_dict = {}
    n_test = X_test.shape[0]
    for step_test in range(0, n_test, 1):
        millis = int(round(time.time() * 1000))
        predict_prob = session.run(
            "predictions/Sigmoid:0", {"input_2:0": X_test[step_test:step_test+1, :, :, :]})
        #label = []
        #preds = []
        prob_value = []
        final_label = []
        #print("enumerate started")
        for i, j in enumerate(predict_prob[0]):
            if j > thresholds[i]:
                prob_value.append(j)
                # preds.append(classes[i])
                # label.append(i)

        #print("comparision started")
        if len(prob_value) >= 1:
            for k in range(0, len(classes)):
                if predict_prob[0][k] == max(prob_value):
                    predict_label = k
                    final_label.append(predict_label)

        else:
            #print("else activated")
            predict_label = predict_prob.argmax()
            final_label.append(predict_label)

        for i in final_label:
            for j in range(len(classes)):
                if i == j:
                    # preds.append(classes[j])
                    global_labels.append(classes[j])
        if verbose:
            print("pred labels : {}, Values : {}, label integer : {} \n".format(
                preds, prob_value, label))
            print("Final label: {} \n".format(final_label))

        if predict_label == Y_test[step_test:step_test + 1]:
            test_acc += 1
        #else:
            # print(predict_label)# predict_label != Y_test[step_test:step_test + 1]:
            # print(X_test.shape)
            # print(step_test+1, predict_label+1)   ## numberical order of images - label
            # cv2.imshow(classes[predict_label] + " <-- " + clAss,X_test[step_test,:,:,:] )
            # cv2.waitKey()
            # print(dirname + classes[predict_label] + " _from_ " + clAss +".jpg")
            # cv2.imwrite(dirname + classes[predict_label] + " _from_ " + clAss + "_imagenum" +str(step_test+1) + ".bmp", X_test[step_test,:,:,:]*256 )
            

        # else:
        #    print(file_test_images[step_test]+"\n")

        label_dict = {i: global_labels.count(i) for i in global_labels}

        curr_millis = int(round(time.time() * 1000))
        time_diff += (curr_millis - millis)

    print(label_dict)
    print(test_acc)
    print(n_test)
    print("Accuracy: %.4f" % ((test_acc*100)/n_test))
    print("Time taken: %i ms." % time_diff)


# Test function for Tensorflow model prediction
def main():
    count = 1
    file_test_images = glob.glob(path_test_images)
    print("Total images {}: ".format(len(file_test_images)))
    no_test_imgs = len(file_test_images)

    X_test = np.zeros([no_test_imgs, height, width, 1], np.float32)
    Y_test = np.zeros(no_test_imgs, np.float32)

    index = 0
    for img_path in file_test_images:
        img = np.array(Image.open(img_path))
        img = scipy.misc.imresize(img, (height, width))
        img = img.reshape((height, width, -1))
        X_test[index] = img.astype(np.float32)

        for i in range(len(classes)):
            if classes[i] in img_path:
                Y_test[index] = i
                break
        index = index + 1

        #count +=1
        sys.stdout.write("\rTotal images processed : %r" % count)
        sys.stdout.flush()

        while count <= no_test_imgs:
            count += 1
            break

    # X_test = (X_test - mean) * scale
    # NOTE: Omit mean
    X_test = (X_test) * scale

    session = tf.Session(graph=load_model())        

    predict(session, X_test, Y_test, file_test_images, verbose=0)


if __name__ == "__main__":
     for clas in classes:
        path_test_images = "/media/aimonsters/Seagate5TB/FCI/512_Augmentation/B1_Pos1/Train/{}/*.bmp".format(clas)
        #clAss = clas
        main()
