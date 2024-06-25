# -*- coding: utf-8 -*-

# @Time    : 2024/6/14 5:09 下午
# @Author  : 张暐（zhangwei）
# @File    : multi_task.py
# @Email   : cosmoszhang@mail.nankai.edu.cn
# @Software: PyCharm

from manim import *
import numpy as np
import random
import sys
import os

sys.path.append('..')

from CS_learning.common_func import CommonFunc

from sklearn.datasets import make_blobs, make_circles,make_moons
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import cluster, datasets


class nlp_task(Scene):
    def construct(self):
        pass

    def nlp_cls(self):
        pass