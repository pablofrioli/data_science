#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from joblib import load
from sklearn.svm import SVC

class Svm_KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier_svm.joblib',
    ):
        self.model = load(model_path)
        
    def __call__(
        self,
        landmark_list,
    ):
        result_index = self.model.predict([np.array(landmark_list).reshape(1,-1)[0]])[0]
        
        return result_index
