#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
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
    
    
    
    
    
1* Training Fusion model
2* GradCAM
'''


''' Reporting results: In[4]
1- Shows Confusion matrix figure
    - you can save it after opening the figure (save figure)
 
2- Generates the classification report

3- Computes & plots the ROC curve and AUC for each class
    - you can save it after opening the figure (save figure)
    
4- Creates a DataFrame to store the true classes and predicted probabilities 
   and saves it to an Excel file
  
    
** This script uses the model's weight (need to be loaded)

'''

import tensorflow as tf
from tensorflow import keras
import cv2
import imutils
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image as preprocessing
import matplotlib.font_manager as fm
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Layer, SeparableConv2D, MaxPool2D, Concatenate, BatchNormalization, ReLU, SpatialDropout2D, Add, MaxPooling2D
import tensorflow as tf
from PIL import Image
import tensorflow.keras
import os
import glob
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Input, Dense, Conv2D, Add, DepthwiseConv2D
from tensorflow.keras.layers import SeparableConv2D, ReLU, SpatialDropout2D
from tensorflow.keras.layers import BatchNormalization, MaxPool2D
from tensorflow.keras.layers import GlobalAvgPool2D, GlobalAveragePooling1D, Average
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support as score
import itertools
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import preprocessing
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2DTranspose
from tensorflow.keras import optimizers
from sklearn.model_selection import KFold
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Add, DepthwiseConv2D
from tensorflow.keras.layers import SeparableConv2D, ReLU, SpatialDropout2D
from tensorflow.keras.layers import BatchNormalization, MaxPool2D
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras import Model
from argparse import Namespace
from IPython.display import Image, display
import matplotlib.cm as cm
from tensorflow.keras import layers
from tensorflow.keras import models
from matplotlib.font_manager import FontProperties
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from vit_keras import vit, utils

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# To ensure reproducibility and minimize variability in training results
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# Clear session before starting model creation
tf.keras.backend.clear_session()

# In[0] Loading data, batch_size, img_size, num_classes
batch_size=16
img_size = (224,224)

# Define the number of classes
num_classes = 7

model_dir_name = 'Ensemble6'

# Main path for the classification directory
main_path = "/HAM10000"
train_path = os.path.join(main_path, 'train')
val_path   = os.path.join(main_path, 'val')

train_datagen = ImageDataGenerator(
      rescale=1./255,
      )
validation_datagen = ImageDataGenerator(rescale=1./255 )

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        )
validation_generator = validation_datagen.flow_from_directory(
        val_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle= False
        )

# In[1] Define the model structure

from tensorflow.keras.layers import Input, Dense, UpSampling2D, SeparableConv2D, BatchNormalization, GlobalAvgPool2D, Reshape, Lambda
from tensorflow.keras.applications import DenseNet201, ResNet50V2
from vit_keras import vit

# Define the input layer
input_shape = (224, 224, 3)
visible = Input(shape=input_shape)

# DenseNet201 model
model1 = DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)
x1 = model1(visible)
x1 = SeparableConv2D(filters=2048, kernel_size=3, padding='same', use_bias=False)(x1)
x1 = BatchNormalization()(x1)

# ResNet50V2 model
model2 = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
x2 = model2(visible)
x2 = SeparableConv2D(filters=2048, kernel_size=3, padding='same', use_bias=False)(x2)
x2 = BatchNormalization()(x2)

# Vision Transformer (ViT) model
model3 = vit.vit_b16(
    image_size=(224, 224),
    activation='softmax', #sigmoid (best than softmax)
    pretrained=True,
    include_top=False,
    pretrained_top=False,
    classes=num_classes
)
model3 = Model(inputs=model3.input, outputs=model3.layers[-2].output)

model3_output = model3(visible)
print(model3_output.shape)

# Custom Layer to remove the first element of the second dimension
class CustomLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        outputs = inputs[:, 1:, :]
        tok = inputs[:, 0, :]
        return outputs, tok

model3_output, tok = CustomLayer()(model3_output)
# Reshape the output to (None, 14, 14, 768)
x3 = Reshape((14, 14, 768))(model3_output)
x3 = MaxPooling2D(pool_size=(2, 2))(x3)

# Combine the ViT output with the combined tensor
combined = tf.concat([x1, x2, x3], axis=3)  # (None, 14, 14, 2816)

# Add classification layers
x = GlobalAvgPool2D()(combined)
x = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=visible, outputs=x)

# Print the model summary
model.summary()



# In[2] Training process

from tensorflow.keras.optimizers.legacy import Adam

my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='Ensemble6_.{epoch:02d}-{val_acc:.4f}.keras'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-5),
              metrics=['acc','FalseNegatives','TrueNegatives','FalsePositives','TruePositives','Precision','Recall']
)
        
history_Xception = model.fit(
                    train_generator,
                    epochs=50,
                    callbacks=my_callbacks,
                    validation_data=validation_generator
)

History = pd.DataFrame(history_Xception.history) 
# or save to csv: 
History_path = 'Ensemble6_History.csv'
with open(History_path, mode='a') as f:
      History.to_csv(f)


# In[3.1] GradCAM     --Load the target model/weights
                    # --Specify the last_conv_4D_layer

last_4D_layer_name = "separable_conv2d"

print('model_dir_name:', model_dir_name)

model_path = os.path.join(main_model_path, f'{model_dir_name}', 'Ensemble6_.44-0.9057.keras')

model.load_weights(model_path)

model.summary(show_trainable = True)

print('model_dir_name:', model_dir_name)
print('model_file    :', model_path.split('/')[-1])


# In[3.2] GradCAM   --Choose the target images (image) from each class

imgs_source = 'val'

skin_class = 'VASC'  # ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

class_folder_path = f'/Dataset/HAM10000/{imgs_source}/{skin_class}'

# Read all images existed in the class path
def get_image_paths(directory, ext=['.jpg', '.png']):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(e) for e in ext):
                image_paths.append(os.path.join(root, file))
    return image_paths
img_paths = get_image_paths(class_folder_path)

print('\n>>> Number of images in the list:', len(img_paths))
print('model_dir_name:', model_dir_name)
print('model_file    :', model_path.split('/')[-1])
print('imgs_source   :', imgs_source + '/' + skin_class)


# In[3.3] Compute GradCAM and save the images
            # Specify the output_dir_path of the saved heatmap images

from GradCAM_module import generate_heatmap

# Names of the classes in order:
name_classes = list(train_generator.class_indices.keys())

skin_class_and_dataset_type = skin_class + '_' + imgs_source.split('_')[0]

output_dir_path = f'/HAM10000/Results/{model_dir_name}/GradCAM/{skin_class_and_dataset_type}'
os.makedirs(output_dir_path, exist_ok=True)

# Generate and save the heatmaps
generate_heatmap(model, img_paths, output_dir_path, name_classes, img_size, model_name=model_dir_name)


# In[4] Reporting results ...
'''
1- Shows Confusion matrix figure
    - you can save it after opening the figure (save figure)
 
2- Generates the classification report

3- Computes & plots the ROC curve and AUC for each class
    - you can save it after opening the figure (save figure)
    
4- Creates a DataFrame to store the true classes and predicted probabilities 
   and saves it to an Excel file
  
    
** This script uses the model's weight (need to be loaded)

'''
# In[4.0.A] Preperation

''' First: You should run the following cells

    1- In[0] Loading data, batch_size, img_size, num_classes
    
    2- In[1] Define the model structure
    
    3- In[3.1] GradCAM
'''

# Title of CM, and part of ROC curve's tilte
model_name = model_dir_name.split('_')[0]
title=f'{model_name}'

# Define the file path for saving the Excel files
prob_excel_file_path = os.path.join(main_model_path , f"{model_dir_name}/{model_dir_name}_prob.xlsx")
CM_excel_file_path = os.path.join(main_model_path , f"{model_dir_name}/{model_dir_name}_metrics_report.xlsx")


# Print:
print('\nmodel_dir_name:', model_dir_name)
print('model_file    :', model_path.split('/')[-1])

# Get the layer's (name & idx) of the model
print('\n-----------', '\t----------')
print('Layer_idx', '\tLayer_name')
print('-----------', '\t----------')
for idx in range(len(model.layers)):
  print(f'{idx}:', '\t', model.get_layer(index = idx).name)


# In[4.0.B] Inference: make predictions on the validation data using the predict_generator method

''' Obtain predictions from the model '''
sns.set_style("white")
pred = model.predict(validation_generator, verbose=1)
predicted_class_indices_test = np.argmax(pred, axis=1)

''' Obtain true labels '''
true_classes = validation_generator.classes

''' Calculate confusion matrix '''
cm = confusion_matrix(true_classes, predicted_class_indices_test)

''' Define target class names (assuming your validation generator provides class indices) '''
name_classes = list(validation_generator.class_indices.keys())


# In[4.1] plot the confusion matrix (save plot after opening the figure)

from CM_module import plot_confusion_matrix
%matplotlib tk

plot_confusion_matrix(cm,
                      name_classes,
                      title,
                      cmap=None,
                      normalize=False,
                      fontsize=20)


# In[4.2] Generate the metrics (classification report) | save the report in xlsx file

from sklearn.metrics import classification_report

''' Opttion_1: print report on the console '''
report = classification_report(true_classes, predicted_class_indices_test, target_names=name_classes)
print(report)


''' Opttion_2: save the report as xlsx file '''
def save_classification_report(true_classes, predicted_classes, target_names, output_filepath):
    """Generates and saves a classification report to an Excel file, preserving the correct structure."""
    try:
        report_dict = classification_report(true_classes, predicted_classes, target_names=target_names, output_dict=True) # Get report as a dictionary

        # Extract accuracy, macro average, and weighted_avg
        accuracy = report_dict.pop('accuracy')
        macro_avg = report_dict.pop('macro avg')
        weighted_avg = report_dict.pop('weighted avg')
        support = macro_avg['support']  # Get the support value


        df = pd.DataFrame(report_dict).transpose()

        # Insert accuracy row in the correct location
        df.loc['accuracy'] = [None, None, accuracy, support]
        df.loc['macro avg'] = macro_avg
        df.loc['weighted avg'] = weighted_avg


        # Reorder rows to match the original report order
        desired_order = target_names + ['accuracy', 'macro avg', 'weighted avg']  # Updated order
        df = df.reindex(desired_order)


        df.to_excel(output_filepath, index=True)
        print(f"Classification report saved to {output_filepath}")

    except Exception as e:
        print(f"Error saving classification report: {e}")
        
save_classification_report(true_classes, predicted_class_indices_test, name_classes, CM_excel_file_path)


# In[4.3] Compute & plot the ROC curve and AUC for each class

# Set Seaborn style
sns.set_style("whitegrid")

''' Compute ROC curve and AUC for each class '''
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(true_classes, pred[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])
   
''' Plot ROC curves for each class '''
plt.figure()
colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
for i, color in zip(range(num_classes), colors):
    plt.plot(
        fpr[i], tpr[i], color=color, lw=2,
        label = '(AUC = {1:0.2f}) ROC curve of class: {0}'.format(name_classes[i], roc_auc[i])
        )

plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# Increase the size of tick labels on x and y axes
plt.xticks(fontsize=22, fontweight='bold')
plt.yticks(fontsize=22, fontweight='bold')

plt.xlabel('False Positive Rate', fontsize=22, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=22, fontweight='bold')
plt.title(f'Receiver Operating Characteristic (ROC)\n{title} model', fontsize=22, fontweight='bold')

# Create a FontProperties object for the legend
legend_font = FontProperties(weight='bold', size=22)  # Adjust size as needed

# Add legend with bold font and increased font size
plt.legend(loc="lower right", prop=legend_font)
# plt.legend(loc="lower right", fontsize=18, prop={'weight': 'bold'})

plt.tight_layout()  # Ensure the layout fits nicely
plt.show()



# In[4.4] Create a DataFrame to store the true classes and predicted probabilities and saves it to an Excel file

# Create a DataFrame to store the data
data = {'True Classes': true_classes, 'Predicted Classes': predicted_class_indices_test}
for i in range(num_classes):
    data[f'Class {i} Probability'] = pred[:, i]
df = pd.DataFrame(data)

# Write the DataFrame to an Excel file
df.to_excel(prob_excel_file_path, index=False)


# In[5] SHAP values ...

'''
1- Visualize the SHAP values
    - you can save it after opening the figure (save figure)
  
** This script uses the model's weight (need to be loaded)

'''
# In[5.0.A] Preperation

''' First: You should run the following cells

    1- In[0] Loading data, img_size, num_classes
    
    2- In[1] Define the model structure
    
    3- In[3.1] GradCAM   --Load the target model/weights
'''

# In[5.1] SHAP     --Prepare the data to be used in SHAP value computations

import shap

# Function to count the number of files in a directory
def count_files(directory):
    return sum([len(files) for r, d, files in os.walk(directory)])

# Count the number of images in training and validation directories
train_count = count_files(train_path) #100
val_count = 0

# Set batch size to the total number of images
batch_size = train_count + val_count   #100

# Separate features and labels
(x_train, y_train) = next(train_generator)
# (x_test, y_test) = next(validation_generator)

# Display the shape of the data
print("Training Data:")
print("\tx_train shape:", x_train.shape)
print("\ty_train shape:", y_train.shape)

print('\nimage type:', train_path.split('/')[-1], '&&', val_path.split('/')[-1])

# Extrac the Name of each class in order
class_names = list(train_generator.class_indices.keys())

# In[5.2] Create an explainer object using SHAP

# # Use DeepExplainer to explain predictions of the model
explainer = shap.GradientExplainer(model, x_train)


# In[5.3] SHAP   --Choose the target images (image) from each class

''' Start:  Use the following lines, when you want to choose from val imgs '''
imgs_source = 'val'

skin_class = 'AKIEC'  # ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

class_folder_path = f'/HAM10000/{imgs_source}/{skin_class}'

# Specify the target image:
img_paths = [
    os.path.join(class_folder_path, 'ISIC_'+'0030408'+'.jpg'),
    
    ]


print('\n>>> Number of images in the list:', len(img_paths))
print('model_dir_name:', model_dir_name)
print('model_file    :', model_path.split('/')[-1])
print('imgs_source   :', imgs_source + '/' + skin_class)


# In[5.4] Visualize the Explanation (Visualize the SHAP Values)

from tensorflow.keras.preprocessing import image

skin_class_and_dataset_type = skin_class + '_' + imgs_source.split('_')[0]
print('imgs_source   :', imgs_source + '/' + skin_class)
output_dir = f'/HAM10000/Results/{model_dir_name}/SHAP/{skin_class_and_dataset_type}'
os.makedirs(output_dir, exist_ok=True)


for img_path in img_paths:
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=img_size)
    img = image.img_to_array(img)
    img /= 255.
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    
    # Calculate SHAP values
    shap_values = explainer.shap_values(img)
    shap_values = np.squeeze(shap_values)
    for class_idx in range(num_classes):
        predictions = model.predict(img)
        
        predicted_class_name = class_names[class_idx]
        
        # Plot the SHAP values
        shap.image_plot(np.expand_dims(shap_values[:,:,:,class_idx], axis=0), img, show=False)
        plt.title(f"{predicted_class_name}", fontsize=20, fontweight='bold', y=1.08)
        plt.xlabel('X Axis', fontsize=22)  # Set the x-axis label font size
        plt.ylabel('Y Axis', fontsize=22)  # Set the y-axis label font size
        plt.xticks(fontsize=22)  # Set tick labels font size
        plt.yticks(fontsize=22)  # Set tick labels font size
        plt.rc('font', size=22)  # Set font size for SHAP details
        # Save the figure
        img_name = os.path.basename(img_path)
        filename = os.path.join(output_dir, img_name.split('.')[0] + f'_{model_dir_name}_{predicted_class_name}_' + '.png')
        plt.savefig(filename, bbox_inches='tight')
        plt.show()
