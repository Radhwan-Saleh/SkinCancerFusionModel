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
    contains the Confusion matrix function
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm,
                          name_classes,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          fontsize=12):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=fontsize+2, fontweight='bold')
    plt.colorbar()

    if name_classes is not None:
        tick_marks = np.arange(len(name_classes))
        plt.xticks(tick_marks, name_classes, rotation=0,
                   fontsize=fontsize, fontweight='bold')
        plt.yticks(tick_marks, name_classes,
                   fontsize=fontsize, fontweight='bold')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=fontsize,
                     fontweight='bold',
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=fontsize,
                     fontweight='bold',
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=fontsize+2, fontweight='bold')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass),
               fontsize=fontsize+2, fontweight='bold')
    plt.show()