# import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D
from tensorflow.keras import applications

# sequential model 
# model = Sequential()
# model.add(applications.VGG16(weights="imagenet"))
# model.summary()

# # functional API
# inp = Input(shape=(224,224,3))
# out = applications.VGG16(weights='imagenet')(inp)
# model = Model(inputs=inp, outputs=out)
# model.summary()

#  Attaching our own classifier
model = Sequential()
model.add(VGG16(include_top=False, input_shape=(224,224,3), pooling='avg'))

model.add(Flatten())
model.add(Dense(10))

model.summary()

# VGG16 is trained on RGB images of size (224, 224), 
# which is a default input size of the network.
# Using a different size, images size should be more than 32 pixels.
# The last dimension which is 3, represents the number of color channels.

# Pooling
# GAP or GMP (global average pooling X global maximum pooling)

# Global pooling: The shape of the output tensor is (3, 3, 512) and
# (7, 7, 512). After applying global pooling on any of these 
# tensors will get us a fixed-size vector of length 512. 
# So the final output of variable size images will still be a
# fixed size vector after applying global pooling.

VGG = VGG16(include_top=False, input_shape=(224,224,3))

for layer in VGG.layers[:10]:
    layer.trainable = False 

for layer in VGG.layers: 
    sp = '     '[len(layer.name)-9:]
    print(layer.name, sp, layer.trainable)

# If the current dataset is similar to the dataset these 
# networks were trained on, then its good to freeze all 
# layers since images in both datasets would have similar 
# features. But if the dataset if different then we should 
# only freeze top layers and train bottom layers because top 
# layers extract general features. More similar the dataset 
# more layers we should freeze.

#  Using one layer of VGG16
model = Sequential()

model.add(Conv2D(64, kernel_size=3, input_shape=(224,224,3)))

model.add(VGG16().layers[2]) # Assing the 3rd layer of pretrained model

model.summary()
