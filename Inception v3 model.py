# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:57:19 2018
This simply uses the inception v3 model to classify images. I got this to work
because I wanted to use the outputs for transfer learning, and this was a
way to get familiar with inception v3. It needs images of size 299x299 and 
values between -1 and 1. The processing steps saved in other code.

You have to give it a file name to classify. There are already GUIs built
around it.

I have a folder named 'panda_pics' just for fun, but I just put whatever
images in it to test out what inception predicts.

A lof of the code snippets are derived from Geron's book:
https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291

@author: Xornack
"""
import tensorflow as tf
from imagenet_class_names import class_names # file I saved.
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim
from skimage import transform as sk_transform
import os
import numpy as np
import matplotlib.pyplot as plt

# Show an image if it's loaded. Saves like a second of time later.
def show(image):
    plt.imshow(image)
    plt.show()
    return None

# Convert to inception  format.
def convert_to_inception_format(image):
    
    # sk_transform.resize actually makes the values between 0 and 1. 
    image = sk_transform.resize(image, (299, 299))
    
    # So I multiplied by 2 and subracted 1 to get values from -1 to 1.
    image = (2*image - np.max(image))/np.max(image)

    return image

# Get a list of images in your folder
def list_image_paths(folder_path, # input the folder containing images.
                     image_formats = ['png', 'jpg', 'jpeg']):
    return [folder_path + '/' + i for i in os.listdir(folder_path) 
            if i.split('.')[-1] in image_formats]

# Set up images.
image_paths = list_image_paths("E:/panda_pics")
unaltered_images = [plt.imread(i) for i in image_paths]
images = [convert_to_inception_format(plt.imread(i)) for i in image_paths]

"""
This code should set up a graph that works with the inception model to read.
Input images and it should make a prediction.
"""
# shape = [batch size, height, width, number of channels].
X = tf.placeholder(tf.float32, shape = [None, 299, 299, 3], name = "X")

# Standard boilerplate for setting up inception from a restore point.
with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_points = inception.inception_v3(
            X, num_classes = 1001, is_training = False)
    predictions = end_points["Predictions"]
    saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # Restore the weights and make a prediction. It's literally that easy.
    # I saved this 'checkpoint' file in it's own folder.
    saver.restore(sess, "E:/inception_v3/inception_v3.ckpt") 
    pred = predictions.eval(feed_dict = {X: images})

# Show all the different classifications.
for i in range(len(unaltered_images)):
    top_5 = np.argpartition(pred[i], -5)[-5:]
    image_preds = [pred[i][j] for j in top_5]
    
    image_guess = []
    for j in range(5):
        # class names is off by 1 from the predictions. The easiest solution
        # is just to subract one from the number. class_names should have an
        # additional value from tf called 'background', and it really should
        # be index 0 in that dictionary.
        image_guess.append((class_names[top_5[j] - 1], image_preds[j]))
        image_guess = sorted(image_guess, key=lambda x: x[1], reverse = True)
     
    show(unaltered_images[i])
    for k in range(5):
        print(image_guess[k][0], image_guess[k][1])
        















