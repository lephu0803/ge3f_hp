from keras.applications.inception_v3 import InceptionV3

base = InceptionV3(weights='imagenet')
print(base.summary())