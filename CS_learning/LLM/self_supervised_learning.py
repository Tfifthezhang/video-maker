# -*- coding: utf-8 -*-

# Copyright (C) 2021 GRGBanking All Rights Reserved

# @Time    : 2023/2/21 5:09 下午
# @Author  : 张暐（zhangwei）
# @File    : maximum_likelihood.py
# @Email   : zhangwei58@grgbanking.com
# @Software: PyCharm

from manim import *
import numpy as np
import sys
import os

sys.path.append('..')

from CS_learning.common_func import CommonFunc

from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.decomposition import KernelPCA
from sklearn import cluster, datasets

class represent_learning(Scene):
    def construct(self):
        self.PCA_example = VGroup()
        #self.intro()
        self.example_PCA()
    def intro(self): # 机器学习的任务是什么，机器学习效果依赖于好的特征
        svg_image = SVGMobject('../images/NN.svg', fill_color=WHITE).scale(1.2)
        self.play(Create(svg_image))
        self.wait(2)
    def example_PCA(self):
        ax = CommonFunc.add_axes(x_range=[-1, 1], y_range=[-1, 1], x_length=8, y_length=6,
                                axis_config={"include_tip": False, "include_numbers": False})
        self.play(Create(ax))
        self.PCA_example.add(ax)

        axes_labels = ax.get_axis_labels(x_label=MathTex('x_1'), y_label=MathTex('x_2'))
        self.play(Create(axes_labels))
        self.PCA_example.add(axes_labels)

        X, y = make_circles(n_samples=300, factor=0.3, noise=0.05, random_state=0)
        coords = list(zip(X[:, 0], X[:, 1], y))
        colors = [BLUE, RED]

        dots = VGroup(
            *[Dot(ax.c2p(coord[0], coord[1]), radius=0.5 * DEFAULT_DOT_RADIUS, color=colors[coord[2]]) for coord in coords])
        self.PCA_example.add(dots)
        self.play(FadeIn(dots))

        self.wait(2)

        ## 引入模型
        self.play(self.PCA_example.animate.scale(0.75).to_edge(UP))
        self.wait(1)

        svg_image = SVGMobject('../images/NN.svg', fill_color=WHITE).scale(0.8).to_edge(1.5*DOWN)
        self.play(Create(svg_image))
        self.wait(2)

        ## 新的坐标架
        kernel_pca = KernelPCA(n_components=2, kernel="rbf", gamma=10, fit_inverse_transform=False, alpha=0.1)
        X = kernel_pca.fit(X).transform(X)
        new_coords = list(zip(X[:, 0], X[:, 1], y))
        new_dots = VGroup(
            *[Dot(ax.c2p(coord[0], coord[1]), radius=0.5 * DEFAULT_DOT_RADIUS, color=colors[coord[2]]) for coord in new_coords])
        new_axes_labels = ax.get_axis_labels(x_label=MathTex('f(x_1)'), y_label=MathTex('f(x_2)'))

        ## 旧坐标变换为新坐标
        self.play(Transform(axes_labels, new_axes_labels))
        self.wait(2)
        self.play(Transform(dots, new_dots))
        self.wait(2)

class supervised_learning(Scene):
    def construct(self):
        self.basic_formula = None
        self.basic_formula_item = {}

        self.write_formula()
        self.unsupervise_example1()

    def write_formula(self):
        formula = MathTex("f", "(", "X", ")", "=", "Y").scale(2.5)
        f, x, y = formula[0], formula[2], formula[-1]
        self.play(Write(formula))
        self.play(x.animate.set_color(BLUE))
        self.wait(1)
        self.play(y.animate.set_color(RED))
        self.wait(1)
        self.play(f.animate.set_color(YELLOW))
        self.wait(2)

        self.basic_formula = formula
        self.basic_formula_item = {'x': x, 'f': f, 'y': y}

        self.play(self.basic_formula.animate.to_edge(UP))
        self.wait(1)

    def unsupervise_example1(self):
        n_samples = 150
        seed = 30
        X, y = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=0)
        kmeans = cluster.KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)

        self.play(Indicate(self.basic_formula_item['x']))

        ax = CommonFunc.add_axes(x_range=[-10, 10], y_range=[-10, 10], x_length=8, y_length=6,
                                 axis_config={"include_tip": False, "include_numbers": False}).scale(0.85).to_edge(DOWN)
        self.play(Create(ax))

        axes_labels = ax.get_axis_labels(x_label=MathTex('x_1'), y_label=MathTex('x_2'))
        self.play(Create(axes_labels))

        kmeans_dots_blue = VGroup(
            *[Dot(ax.c2p(coord[0], coord[1]), radius=0.5 * DEFAULT_DOT_RADIUS, color=WHITE)
             for coord in list(zip(X[:, 0], X[:, 1], kmeans.labels_)) if coord[-1]==2 ])
        kmeans_dots_orange = VGroup(
            *[Dot(ax.c2p(coord[0], coord[1]), radius=0.5 * DEFAULT_DOT_RADIUS, color=WHITE)
             for coord in list(zip(X[:, 0], X[:, 1], kmeans.labels_)) if coord[-1]==0])
        kmeans_dots_green = VGroup(
            *[Dot(ax.c2p(coord[0], coord[1]), radius=0.5 * DEFAULT_DOT_RADIUS, color=WHITE)
             for coord in list(zip(X[:, 0], X[:, 1], kmeans.labels_)) if coord[-1]==1])
        kmeans_dot = VGroup(kmeans_dots_green, kmeans_dots_orange, kmeans_dots_blue)

        self.play(FadeIn(kmeans_dot))
        self.wait(1)

        self.play(kmeans_dots_blue.animate.set_color(BLUE, family=True))
        self.play(kmeans_dots_green.animate.set_color(GREEN, family=True))
        self.play(kmeans_dots_orange.animate.set_color(ORANGE, family=True))
        self.wait(2)


    def supervise_example(self):
        self.play(Indicate(self.basic_formula_item['x']),
                  Indicate(self.basic_formula_item['y']))
        self.wait(1)


class thanks_end(Scene):
    def construct(self):
        svg_image = SVGMobject('svg_icon/bird.svg', fill_color=MAROON).scale(1.2).to_edge(6 * LEFT + UP)

        text = Text('感谢充电', font='SIL-Hei-Med-Jian').scale(1.2).next_to(svg_image, 2.5 * DOWN)

        self.play(SpinInFromNothing(svg_image))

        self.play(Create(text))

        source_path = 'svg_icon/charge/'

        l_image_path = [image_path for image_path in os.listdir(source_path) if image_path.split('.')[-1] == 'jpg']
        vg_anchor = VGroup(*[Circle(1) for _ in range(len(l_image_path))])
        vg_anchor.arrange_in_grid(rows=3, buff=0.1).to_edge(2 * RIGHT)
        # self.add(vg_anchor)

        for i in range(len(l_image_path)):
            image_path = l_image_path[i]
            split_str = image_path.split('.')
            if split_str[-1] == 'jpg':
                name = split_str[0]
                # image = cv2.imread(os.path.join(source_path, image_path),3)
                # resize_array = cv2.resize(image, (512,512))
                image = ImageMobject(os.path.join(source_path, image_path)).move_to(vg_anchor[i].get_center())
                image.height = 1.2
                image.width = 1.2
                name_text = Text(name).scale(0.6).next_to(image, 0.5 * DOWN)
                self.add(image, name_text)

        self.wait(3)
