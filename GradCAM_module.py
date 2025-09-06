#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Article Title: Explainable Depth-Wise and Channel-Wise Fusion Models for Multi-Class Skin Lesion Classification

Authors:
    {
        Humam AbuAlkebash 1
        Radhwan A. A. Saleh 2
        H. Metin Ertunç 1
        Mugahed A. Al-antari 3
    }
    
Affiliations:
    1 Department of Mechatronics Engineering, Kocaeli University, Kocaeli, Türkiye
    2 Department of Software Engineering, Kocaeli University, Kocaeli, Türkiye
    3 Department of Artificial Intelligence, College of Software \& Convergence Technology, Daeyang AI Center, Sejong University, Seoul, South Korea
    
    
    
    
description:
    contains the GradCAM class, which has functions to be imported from the model script
"""

import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt



class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()
        print('\nThe Selected layer to be passed to GradCAM class:', self.layerName, '..................................>>>\n')

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            
            loss = predictions[:, tf.argmax(predictions[0])]
    
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_JET):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(255-heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)



def generate_heatmap(model, img_paths, output_dir_path, class_names, img_size=(224,224), model_name='model_name', last_4D_layer_name=None):
    for img_path in img_paths:
        # Prediction
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # to show this image after resizing it
        image = cv2.resize(image, img_size)
        image = image.astype('float32') / 255
        image = np.expand_dims(image, axis=0)
        
        preds = model.predict(image)
        i = np.argmax(preds[0])
        class_pred = class_names[i]
        
        
        # Get the layer's (name & idx) of the model
        # for idx in range(len(model.layers)):
        #   print(f'{idx}:', model.get_layer(index = idx).name)
        
        
        # Passing to GradCAM class
        icam = GradCAM(model, i, last_4D_layer_name)  # we pick the 4D layer OR keep it "None" to let the fcn decide the last_4D_layer to be used
        heatmap = icam.compute_heatmap(image)
        heatmap = cv2.resize(heatmap, img_size)
        
        # Read the image agin to overlap
        image = cv2.imread(img_path)
        image = cv2.resize(image, img_size)
        print(heatmap.shape, image.shape)
        
        # Overlay the heatmap onto the input Image (2_imgs to one image)
        (heatmap1, output) = icam.overlay_heatmap(heatmap, image, alpha=0.5)
        
        
        # Visualization
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(cv2.resize(image_rgb, img_size))
        ax[0].set_title('Original Image')
        ax[1].imshow(heatmap1)
        ax[1].set_title('Heatmap')
        ax[2].imshow(output)
        ax[2].set_title('Overlaid Image')
        
        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.15)
        
        # Save the figure
        img_name = os.path.basename(img_path)
        file_name = img_name.split('.')[0] + '_' + model_name + '_' + class_pred + '_' + str(preds[0][i]) + '.png'
        os.makedirs(output_dir_path, exist_ok=True)
        fig.savefig(os.path.join(output_dir_path, file_name), bbox_inches='tight')
