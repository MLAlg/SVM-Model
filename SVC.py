#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 06:17:23 2019

@author: samaneh
"""

#image recognition --> SVM
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces()
print(faces.DESCR) # description
print(faces.keys())
print(faces.images.shape)
print(faces.data.shape)
print(faces.target.shape)

# no need to normalization            
print(np.min(faces.data))
print(np.max(faces.data))
print(np.mean(faces.data))

#plot some faces
def print_faces(images, target, top_n):
    fig = plt.figure(figsize=(12, 12)) # set up the figure size in inches
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0.05, wspace=0.05)
    for i in range(top_n):
        # plot the images in a matrix of 20x20
        p = fig.add_subplot(20, 20, i+1, xticks=[], yticks=[]) 
        p.imshow(images[i], cmap=plt.cm.bone)
        # label the image with the target value                    
        p.text(0, -12, str(target[i]))
        p.text(0, 90, str(i))
        
print_faces(faces.images, faces.target, 20)

from sklearn.svm import SVC # Support Vector Classifier
svc_1 = SVC(kernel='linear')

from sklearn.model_selection import train_test_split # split dataset
x_train, x_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.25, random_state=0)

from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem # calculate standard error

def evaluate_cross_validation(clf, x, y, k):
    cv = KFold(n_splits=5, shuffle=True, random_state=0) #iterator
    scores = cross_val_score(clf, x, y, cv=cv)
    print(scores)
    return("Mean score: {0:.3f} (+/- {1: .3f})").format(np.mean(scores), sem(scores))
    
evaluate_cross_validation(svc_1, x_train, y_train, 5)

from sklearn import metrics
def train_and_evaluate(clf, x_train, x_test, y_train, y_test):
    clf.fit(x_train, y_train)
    print("Accuracy on training set: ")
    print(clf.score(x_train, y_train))
    print("Accuracy on testing set: ")
    print(clf.score(x_test, y_test)) 
    y_pred = clf.predict(x_test)
    print("Classification Report: ")
    print(metrics.classification_report(y_test, y_pred))
    print("Confusion Matrix: ")
    print(metrics.confusion_matrix(y_test, y_pred))
    
train_and_evaluate(svc_1, x_train, x_test, y_train, y_test)    

glasses = [
      (10, 19), (30, 32), (37, 38), (50, 59), (63, 64),
      (69, 69), (120, 121), (124, 129), (130, 139), (160, 161),
      (164, 169), (180, 182), (185, 185), (189, 189), (190, 192),
      (194, 194), (196, 199), (260, 269), (270, 279), (300, 309),
      (330, 339), (358, 359), (360, 369)
 ]
def create_target(segments):
    y = np.zeros(faces.target.shape[0]) # create a new y array of target size initialized with zeros
    for (start, end) in segments: # # put 1 in specified segments
        y[start: end+1] = 1
    return y
target_glasses = create_target(glasses)

x_train, x_test, y_train, y_test = train_test_split(faces.data, target_glasses, test_size=0.25, random_state=0)
svc_2 = SVC(kernel='linear')
evaluate_cross_validation(svc_2, x_train, y_train, 5)
train_and_evaluate(svc_2, x_train, x_test, y_train, y_test)































    