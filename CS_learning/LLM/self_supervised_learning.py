# -*- coding: utf-8 -*-

# Copyright (C) 2021 GRGBanking All Rights Reserved

# @Time    : 2023/2/21 5:09 下午
# @Author  : 张暐（zhangwei）
# @File    : maximum_likelihood.py
# @Email   : zhangwei58@grgbanking.com
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


class represent_learning(Scene):
    def construct(self):
        self.PCA_example = VGroup()
        self.NN_func = VGroup()
        self.NN_node = None
        self.NN_weight = None
        self.cat_dog_example = VGroup()

        self.linear_cant()
        self.feature_trans()
        self.nn()
        self.nn_structure()
        self.nn_example1()
        self.represent_clf()

    def intro(self):  # 机器学习的任务是什么，机器学习效果依赖于好的特征
        svg_image = SVGMobject('../images/NN.svg', fill_color=WHITE).scale(1.2)
        self.play(Create(svg_image))
        self.wait(2)

    def linear_cant(self):
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
            *[Dot(ax.c2p(coord[0], coord[1]), radius=0.5 * DEFAULT_DOT_RADIUS, color=colors[coord[2]]) for coord in
              coords])
        self.PCA_example.add(dots)
        self.play(FadeIn(dots))

        self.wait(2)

        ## 不是线性可分的
        graph = ax.plot(lambda x: x + 0.5, x_range=[-0.8, 0.5], use_smoothing=True, color=YELLOW)
        self.play(Create(graph))
        self.play(Rotate(graph, angle=2 * PI,
                         about_point=ax.c2p(0, 0),
                         rate_func=linear, run_time=2))
        self.wait(2)
        self.play(FadeOut(graph))
        ## 引入模型
        self.play(self.PCA_example.animate.scale(0.75).to_edge(UP))
        self.wait(1)

    def feature_trans(self):
        ax, axes_labels, dots = self.PCA_example
        svg_image = SVGMobject('../images/NN.svg', fill_color=WHITE).scale(0.8).to_edge(2 * DOWN)
        self.play(Create(svg_image))
        self.wait(2)

        self.NN_func.add(svg_image)

        formula_x = MathTex("X").scale(2).next_to(svg_image, 2 * LEFT)

        self.play(FadeTransform(axes_labels, formula_x))
        self.wait(1)

        self.NN_func.add(formula_x)

        ## X进入模型变为f（x）

        self.play(FadeOut(formula_x.copy(), target_position=svg_image, run_time=1))

        formula_y = MathTex("f(X)").scale(2).next_to(svg_image, 2 * RIGHT)

        self.play(FadeIn(formula_y, target_position=svg_image, run_time=1))

        self.wait(2)

        self.NN_func.add(formula_y)

        ## 新的坐标架
        X, y = make_circles(n_samples=300, factor=0.3, noise=0.05, random_state=0)
        colors = [BLUE, RED]
        kernel_pca = KernelPCA(n_components=2, kernel="rbf", gamma=10, fit_inverse_transform=False, alpha=0.1)
        X = kernel_pca.fit(X).transform(X)
        new_coords = list(zip(X[:, 0], X[:, 1], y))
        new_dots = VGroup(
            *[Dot(ax.c2p(coord[0], coord[1]), radius=0.5 * DEFAULT_DOT_RADIUS, color=colors[coord[2]]) for coord in
              new_coords])
        new_axes_labels = ax.get_axis_labels(x_label=MathTex('f(x_1)'), y_label=MathTex('f(x_2)'))

        ## 坐标架更换
        self.play(FadeTransform(formula_y.copy(), new_axes_labels))
        self.wait(2)
        ## 旧坐标变换为新坐标
        self.play(Transform(dots, new_dots))
        self.wait(2)

        self.PCA_example.add(new_axes_labels)

        ## 线性可分
        graph = ax.plot(lambda x: -0.5 * x - 0.2, x_range=[-1, 1], use_smoothing=True, color=YELLOW)
        self.play(SpinInFromNothing(graph))

        self.wait(2)

        self.PCA_example.add(graph)

        self.play(FadeOut(self.PCA_example))
        self.wait(2)

    def nn(self):
        def get_nodes(n):
            nodes = VGroup(*[Circle(radius=0.23
                                    , stroke_color=BLUE
                                    , stroke_width=2
                                    , fill_color=GRAY
                                    , fill_opacity=0
                                    ) for _ in range(n)])
            nodes.arrange(DOWN, buff=0.2)
            return nodes
        node1 = get_nodes(6)
        node2 = get_nodes(6)
        node3 = get_nodes(6)

        vg_nodes = VGroup(node1, node2, node3)

        vg_nodes.arrange(RIGHT, buff=1).next_to(self.NN_func[0], UP)

        self.play(FadeIn(vg_nodes, target_position=self.NN_func[0]))

        self.wait(2)

        self.NN_node = vg_nodes

        def create_connections(left_layer_nodes, right_layer_nodes):
            # Create VGroup to hold created connections
            connection_group = VGroup()
            # Iterate through right layer nodes
            for l in range(len(right_layer_nodes)):
                # Iterate through left layer nodes
                for r in range(len(left_layer_nodes)):
                    # Create connection line
                    line = Line(start=right_layer_nodes[l].get_edge_center(LEFT)
                                , end=left_layer_nodes[r].get_edge_center(RIGHT)
                                , color=WHITE,
                                stroke_width=0.75
                                # , stroke_opacity=0.4
                                )

                    # Add to connection group
                    connection_group.add(line)
            return connection_group

        weight_12 = create_connections(vg_nodes[0], vg_nodes[1])
        weight_23 = create_connections(vg_nodes[1], vg_nodes[2])

        self.play(Create(weight_12), Create(weight_23))
        self.wait(2)

        self.NN_weight = VGroup(weight_12, weight_23)

    def nn_structure(self):
        vg_nn = VGroup(self.NN_node, self.NN_weight)

        cnn = SVGMobject('svg_icon/cnn.svg',
                         opacity=None,
                         stroke_color=BLUE,
                         stroke_opacity=1,
                         stroke_width=1).scale(0.8).to_edge(RIGHT).shift(1.5*UP)

        self.play(FadeTransform(vg_nn.copy(), cnn))
        self.wait(1)

        rnn = SVGMobject('svg_icon/rnn.svg',
                         opacity=None,
                         stroke_color=BLUE,
                         stroke_opacity=1,
                         stroke_width=1).scale(1).next_to(cnn, 2*DOWN)

        self.play(FadeTransform(vg_nn.copy(), rnn))
        self.wait(1)

        transformer = SVGMobject('svg_icon/attention.svg',
                                 opacity=None,
                                 stroke_color=BLUE,
                                 stroke_opacity=0.9,
                                 stroke_width=0.8).scale(1.3).to_edge(0.5*LEFT)

        self.play(FadeTransform(vg_nn.copy(), transformer))
        self.wait(1)

        self.play(FadeOut(cnn, target_mobject=vg_nn),
                  FadeOut(rnn, target_mobject=vg_nn),
                  FadeOut(transformer, target_mobject=vg_nn))
        self.wait(1)

    def nn_example1(self):
        # self.NN_func
        cat = SVGMobject('svg_icon/cat.svg').scale(0.6).to_edge(LEFT+UP)
        dog = SVGMobject('svg_icon/dog.svg').scale(0.6).next_to(cat, RIGHT)

        self.play(SpinInFromNothing(cat),
                  SpinInFromNothing(dog))

        self.wait(2)

        self.cat_dog_example.add(VGroup(cat, dog))

        ax = CommonFunc.add_axes(x_range=[-6, 7], y_range=[-7, 6], x_length=6, y_length=6,
                         axis_config={"include_tip": False, "include_numbers": False}).scale(0.6).next_to(self.NN_node[0],3*LEFT).shift(DOWN)

        # axes_labels = ax.get_axis_labels(x_label=MathTex('x_1'), y_label=MathTex('x_2'))
        # self.cat_dog_example.add(axes_labels)

        X, y = make_moons(n_samples=30, noise=2.5, random_state=0)
        coords = list(zip(X[:, 0], X[:, 1], y))

        dots = VGroup(
            *[cat.copy().scale(0.4).move_to(ax.c2p(coord[0], coord[1])) if coord[-1] == 1
             else dog.copy().scale(0.4).move_to(ax.c2p(coord[0], coord[1]))
             for coord in coords])
        self.cat_dog_example.add(VGroup(ax, dots))
        self.play(FadeIn(self.cat_dog_example[1:], target_position=self.cat_dog_example[0]))

        self.wait(2)

        self.play(FadeOut(self.cat_dog_example[1:].copy(), target_position=self.NN_node[0]))

        ## 不是线性可分的
        #graph = ax.plot(lambda x: x + 0.5, x_range=[-0.8, 0.5], use_smoothing=True, color=YELLOW)


        self.play(Indicate(self.NN_node[0]))
        #self.play(ShowPassingFlash(self.NN_weight[0].copy().set_color(YELLOW)))
        self.play(self.NN_weight[0].animate.set_color(YELLOW))
        self.play(Indicate(self.NN_node[1]))
        #self.play(ShowPassingFlash(self.NN_weight[1].copy().set_color(YELLOW)))
        self.play(self.NN_weight[1].animate.set_color(YELLOW))
        self.play(Indicate(self.NN_node[2]))
        
        bx = CommonFunc.add_axes(x_range=[-2, 3.5], y_range=[-1, 6], x_length=6, y_length=6,
                         axis_config={"include_tip": False, "include_numbers": False}).scale(0.6).next_to(
            self.NN_node[2], 3 * RIGHT).shift(DOWN)

        X, y = make_blobs(n_samples=30, centers=2, n_features=2, cluster_std=0.8, random_state=0)
        coords = list(zip(X[:, 0], X[:, 1], y))

        dots = VGroup(
            *[cat.copy().scale(0.4).move_to(bx.c2p(coord[0], coord[1])) if coord[-1] == 1
              else dog.copy().scale(0.4).move_to(bx.c2p(coord[0], coord[1]))
              for coord in coords])
        self.play(FadeIn(VGroup(bx, dots), target_position=self.NN_node[-1]))

        self.wait(2)

        self.cat_dog_example.add(VGroup(bx, dots))

    def represent_clf(self):
        def get_nodes(n):
            nodes = VGroup(*[Circle(radius=0.4
                                    , stroke_color=RED
                                    , stroke_width=2
                                    , fill_color=GRAY
                                    , fill_opacity=0
                                    ) for _ in range(n)])
            nodes.arrange(DOWN, buff=0.2)
            return nodes
        def create_connections(left_layer_nodes, right_layer_nodes):
            # Create VGroup to hold created connections
            connection_group = VGroup()
            # Iterate through right layer nodes
            for l in range(len(right_layer_nodes)):
                # Iterate through left layer nodes
                for r in range(len(left_layer_nodes)):
                    # Create connection line
                    line = Line(start=right_layer_nodes[l].get_edge_center(LEFT)
                                , end=left_layer_nodes[r].get_edge_center(RIGHT)
                                , color=WHITE,
                                stroke_width=1)
                    connection_group.add(line)
            return connection_group

        def sigmoid(x):
            s = np.power(np.e, -x)
            return 1 / (1 + s)

        self.play(FadeOut(self.cat_dog_example[0]),
                  FadeOut(self.cat_dog_example[1]),
                  self.NN_func.animate.to_edge(LEFT),
                  VGroup(self.NN_node, self.NN_weight).animate.to_edge(LEFT),
                  self.cat_dog_example[-1].animate.to_edge(RIGHT).shift(UP))

        self.wait(2)

        # 输出y
        svg_clf = SVGMobject('svg_icon/logistic.svg', fill_color=WHITE).scale(0.6).next_to(self.NN_func[-1], RIGHT)
        self.NN_func.add(svg_clf)

        self.play(FadeIn(svg_clf))
        self.play(FadeOut(self.NN_func[-1].copy(), target_position=svg_clf))

        clf = get_nodes(1)
        clf.shift(0.8*UP)

        self.play(FadeIn(clf, target_position=svg_clf))

        weight_output = create_connections(self.NN_node[-1], clf)

        self.play(Create(weight_output))

        clf_brace = Brace(clf, DOWN)
        clf_text = Text('输出层', color=MAROON).scale(0.8).next_to(clf_brace, DOWN)

        self.play(Write(clf_brace))
        self.play(Write(clf_text))

        ## 输出层可以为sigmoid

        ax = CommonFunc.add_axes(x_range=[-8, 8], y_range=[0, 1], x_length=7, y_length=4,
                                 axis_config={"include_tip": False, "include_numbers": True})

        sigmoid_plot = ax.plot(lambda x: sigmoid(x), x_range=[-7, 7], use_smoothing=True, color=MAROON)
        vg_sigmoid = VGroup(ax, sigmoid_plot).scale(0.35).next_to(clf, UP)

        self.play(FadeTransform(clf.copy(), vg_sigmoid))
        self.wait(2)

        # 决策边界
        # from sklearn.linear_model import LogisticRegression
        X, y = make_blobs(n_samples=30, centers=2, n_features=2, cluster_std=0.8, random_state=0)
        log_clf = LogisticRegression(random_state=0)
        log_clf.fit(X, y)
        a1, a2 = log_clf.coef_[0]
        b = log_clf.intercept_[0]
        graph = self.cat_dog_example[-1][0].plot(lambda x: -(a1 / a2) * x - b / a2, x_range=[-2.5, 3.5], use_smoothing=True, color=MAROON)


        Y_tex = MathTex('Y', color=MAROON).scale(2).next_to(svg_clf, 2*RIGHT)
        self.NN_func.add(Y_tex)

        self.play(FadeIn(Y_tex, target_position=svg_clf),
                  FadeTransform(clf.copy(), graph))
        self.wait(2)

        self.play(Indicate(Y_tex),
                  Indicate(graph))
        self.wait(2)

        self.play(FadeOut(VGroup(self.NN_weight, self.NN_node,
                                 self.cat_dog_example[-1], graph,
                                 clf, weight_output, clf_text, clf_brace)),
                  self.NN_func.animate.to_edge(UP))

    def represent_convert(self):
        ax = NumberPlane(x_range=[-6, 6], y_range=[-4, 4], x_length=8, y_length=6,
                         axis_config={"include_tip": False, "include_numbers": False}).to_edge(RIGHT)
        arrow = Arrow(ORIGIN, [2, 2, 0], buff=0)
        tip_text = Text('(2, 2)').next_to(arrow.get_end(), RIGHT)





class unsupervised_example(Scene):
    def construct(self):
        self.data_image = None
        self.example_ax = VGroup()

        self.unsupervised_intro()
        self.data_prepare()
        self.kmeans_run()

    def unsupervised_intro(self):
        unsup_cn = Text('无监督学习').to_edge(LEFT + 2 * UP)
        unsup_en = Text('Unsupervised Learning').scale(0.5).next_to(unsup_cn, DOWN)

        self.play(FadeIn(unsup_cn), FadeIn(unsup_en))

        svg_image = SVGMobject('svg_icon/文档.svg', fill_color=WHITE).scale(0.9).next_to(unsup_en, 2 * DOWN)
        text_intro = Text('数据无标注', color=GRAY).scale(0.7).next_to(svg_image, DOWN)

        image_text_intro = VGroup(svg_image, text_intro)

        self.play(Create(image_text_intro))
        self.wait(2)

        self.data_image = image_text_intro

    def data_prepare(self):
        centers = [[-2, -2], [0, 3], [3, 0]]
        X, y = make_blobs(n_samples=18, centers=centers, cluster_std=1, random_state=0)
        clf = LogisticRegression(random_state=0,
                                 multi_class='ovr',
                                 solver='newton-cg')
        clf.fit(X, y)

        ax = NumberPlane(x_range=[-6, 6], y_range=[-4, 4], x_length=8, y_length=6,
                         axis_config={"include_tip": False, "include_numbers": False}).to_edge(RIGHT)
        self.example_ax.add(ax)

        coords = list(zip(X[:, 0], X[:, 1], y))
        colors = [BLUE, RED, GREEN]

        dots = VGroup(
            *[Dot(ax.c2p(coord[0], coord[1]),
                  radius=1 * DEFAULT_DOT_RADIUS,
                  stroke_width=8,
                  fill_opacity=.7,
                  color=WHITE) for coord in coords])
        self.example_ax.add(dots)

        init_centers = [[-3, -3], [-2, 2], [3, -2]]
        centroids = VGroup(*[Dot(ax.c2p(center[0], center[1]),
                                 color=colors[i],
                                 radius=3 * DEFAULT_DOT_RADIUS,
                                 stroke_width=1,
                                 fill_opacity=.9) for i, center in enumerate(init_centers)])

        self.play(FadeTransform(self.data_image[1].copy(), self.example_ax))
        self.wait(2)

        self.play(Create(centroids), run_time=2)
        self.wait(2)
        self.example_ax.add(centroids)

    @staticmethod
    def stuff_sorted_by_distance(x, stuff):
        x_np = np.array(x.get_center())
        return sorted(stuff, key=lambda i: np.linalg.norm(x_np - np.array(i.get_center())))

    def distance_to_closest_centroid(self, dot, centroids):
        closest_centroid = np.array(self.stuff_sorted_by_distance(dot, centroids)[0].get_center())
        return np.linalg.norm(np.array(dot.get_center()) - closest_centroid)

    def kmeans_run(self):
        ax, dots, centroids = self.example_ax
        n_runs = 3

        for run_idx in range(n_runs):
            self.wait(1.6)
            # === Get closest centroid for each datapoint ===
            for dot in dots:
                closest_centroid = self.stuff_sorted_by_distance(dot, centroids)[0]
                my_color = closest_centroid.get_color()

                radius_tracker = ValueTracker(0)
                growing_circle = always_redraw(
                    lambda: Circle(radius=radius_tracker.get_value(), color=WHITE, stroke_width=2).move_to(
                        dot.get_center()))
                self.add(growing_circle)

                self.play(
                    radius_tracker.animate.set_value(self.distance_to_closest_centroid(dot, centroids)),
                    FadeIn(Line(closest_centroid.get_center(), dot.get_center(), color=my_color, buff=.2),
                           rate_func=there_and_back),
                    dot.animate.set_color(my_color),
                )
                self.remove(growing_circle)

            # === Move each centroid to mean ===
            for centroid in centroids:
                relevant_dots = list(filter(lambda d: d.get_color() == centroid.get_color(), dots))
                new_pos = np.array([d.get_center() for d in relevant_dots]).mean(axis=0)
                self.play(centroid.animate.move_to(new_pos), run_time=1.3)
        self.wait(2)


class supervised_example(ThreeDScene):
    def construct(self):
        self.clf = None
        self.example_ax = VGroup()
        self.data_image = None
        self.text = VGroup()
        self.ax_3D = VGroup()
        self.m = 10
        self.loss_track = None
        self.loss_graph = None
        self.iterations = None
        self.learning_rate = [0.4]

        self.supervised_intro()
        self.data_prepare()
        self.write_text_before_animate()
        self.get_surfaceplot()
        self.get_dot()

    def supervised_intro(self):
        unsup_cn = Text('有监督学习').to_edge(LEFT + 2 * UP)
        unsup_en = Text('supervised Learning').scale(0.5).next_to(unsup_cn, DOWN)

        self.play(FadeIn(unsup_cn), FadeIn(unsup_en))

        svg_image = SVGMobject('svg_icon/文档.svg', fill_color=WHITE).scale(0.9).next_to(unsup_en, 2 * DOWN)
        text_intro = Text('数据有标注', color=GRAY).scale(0.7).next_to(svg_image, DOWN)

        l_label_svg = VGroup(
            *[SVGMobject('svg_icon/文档_split.svg', fill_color=i).scale(0.6) for i in [BLUE, RED, GREEN]])
        l_label_svg.arrange_submobjects(RIGHT).scale(0.8).next_to(text_intro, DOWN)

        image_text_intro = VGroup(svg_image, text_intro)
        self.data_image = image_text_intro

        self.play(Create(image_text_intro))
        self.wait(2)

        self.play(FadeTransform(text_intro.copy(), l_label_svg))

        self.data_image.add(l_label_svg)

    def data_prepare(self):
        centers = [[-2, -2], [0, 3], [3, 0]]
        X, y = make_blobs(n_samples=18, centers=centers, cluster_std=1, random_state=0)
        clf = LogisticRegression(random_state=0,
                                 multi_class='ovr',
                                 solver='newton-cg')
        clf.fit(X, y)

        self.clf = clf

        ax = NumberPlane(x_range=[-6, 6], y_range=[-4, 4], x_length=8, y_length=6,
                         axis_config={"include_tip": False, "include_numbers": False}).to_edge(RIGHT)
        self.example_ax.add(ax)

        coords = list(zip(X[:, 0], X[:, 1], y))
        colors = [BLUE, RED, GREEN]

        dots = VGroup(
            *[Dot(ax.c2p(coord[0], coord[1]),
                  radius=1 * DEFAULT_DOT_RADIUS,
                  stroke_width=8,
                  fill_opacity=.7, color=colors[coord[2]]) for coord in coords])
        self.example_ax.add(dots)

        self.play(FadeIn(self.example_ax))
        self.wait(2)

        l_graph = VGroup()
        for coefs, intercept in zip(clf.coef_, clf.intercept_):
            a1, a2 = coefs
            b = intercept
            graph = ax.plot(lambda x: -(a1 / a2) * x - b / a2, x_range=[-6, 6], use_smoothing=True, color=YELLOW_C)
            l_graph.add(graph)

        self.example_ax.add(l_graph)
        self.play(FadeIn(l_graph))
        self.wait(2)

    def write_text_before_animate(self):
        self.play(FadeOut(self.data_image[0]),
                  FadeOut(self.data_image[-1]),
                  FadeOut(self.example_ax[-1]),
                  self.data_image[1].animate.shift(2.5 * UP))
        self.wait(1)

        formula_x = MathTex("X").scale(1).next_to(self.data_image[1], 2.5 * DOWN)
        self.text.add(formula_x)
        self.play(Create(formula_x))
        self.wait(1)

        svg_image = SVGMobject('../images/NN.svg', fill_color=WHITE).scale(0.6).next_to(formula_x, 2 * RIGHT)
        self.text.add(svg_image)
        self.play(FadeIn(svg_image))
        self.wait(2)

        self.play(FadeOut(formula_x.copy(), target_position=svg_image, run_time=1))
        formula_y = MathTex("Y_{pred}").scale(1).next_to(svg_image, 2 * RIGHT)

        self.play(FadeIn(formula_y, target_position=svg_image, run_time=1))

        self.wait(2)

        self.play(Indicate(self.data_image[1]))
        self.wait(1)
        y_ture = MathTex("Y_{true}").scale(1).next_to(formula_y, DOWN)
        self.play(FadeTransform(self.data_image[1].copy(), y_ture))
        self.wait(2)

        text_loss = MathTex(" \mathcal{L}(Y_{true},Y_{pred})").next_to(formula_x, 3 * DOWN)
        self.text.add(text_loss)

        self.play(FadeTransform(VGroup(y_ture, formula_y), text_loss))

        self.wait(2)

        g_text = MathTex('\mathbf{g} = \\nabla \mathcal{L}').scale(0.8).next_to(text_loss, DOWN)

        self.text.add(g_text)

        text_gradient = MathTex("\mathbf{\omega}_{m+1} =",
                                "\mathbf{\omega}_m - ",
                                "\epsilon", "\mathbf{g}(\omega_m)").scale(0.6).next_to(g_text, DOWN)
        self.text.add(text_gradient)

        self.play(Write(text_gradient), Write(g_text))
        self.wait(2)

    def get_surfaceplot(self):
        # self.play(self.example_ax.animate.scale().to_edge(RIGHT))
        self.play(FadeOut(self.text))

        def bowl(u, v):
            z = 1 / 2 * (np.power(u, 2) + np.power(v, 2))
            return z

        axes = ThreeDAxes(x_range=(-4, 4), y_range=(-4, 4), z_range=(0, 7), x_length=8, y_length=8, z_length=4).scale(
            0.85).to_edge(0.5 * LEFT)
        self.ax_3D.add(axes)
        surface_plane = Surface(lambda u, v: axes.c2p(u, v, bowl(u, v)),
                                u_range=[-2, 2],
                                v_range=[-2, 2],
                                resolution=(30, 30),
                                should_make_jagged=True,
                                stroke_width=0.2)

        surface_plane.set_style(fill_opacity=0.5, stroke_color=RED)

        surface_plane.set_fill_by_value(axes=axes, colors=[(RED, 0.0), (YELLOW, 0.2), (BLUE, 4)], axis=2)

        self.ax_3D.add(surface_plane)

        self.play(Create(self.ax_3D))

        self.play(Rotate(self.ax_3D, angle=-90 * DEGREES, axis=RIGHT))
        self.wait(2)

        # self.move_camera(phi=75 * DEGREES)

        # self.set_camera_orientation(phi=75 * DEGREES, theta=0)

        # for i in range(6):
        #      self.play(Rotate(self.ax_3D, angle=60 * DEGREES, axis=UP))

        self.play(Rotate(self.ax_3D, angle=15 * DEGREES, axis=UP))
        self.play(Rotate(self.ax_3D, angle=45 * DEGREES, axis=RIGHT))

    @staticmethod
    def f(x, y):
        z = 1 / 2 * (np.power(x, 2) + np.power(y, 2))
        return np.array([x, y, z])

    def GD(self, lr, start):
        def partial_x(x):
            return x

        def partial_y(y):
            return y

        x, y = start[0], start[1]
        GD_x, GD_y, GD_z = [], [], []
        for it in range(self.m):
            GD_x.append(x)
            GD_y.append(y)
            GD_z.append(self.f(x, y)[-1])
            dx = partial_x(x)
            dy = partial_y(y)
            x = x - lr * dx
            y = y - lr * dy
        return GD_x, GD_y, GD_z

    def get_dot(self):
        ax = self.example_ax[0]
        iter_graphs = VGroup()
        for extra_bias in np.linspace(10, 1, 10):
            l_graph = VGroup()
            for coefs, intercept in zip(self.clf.coef_, self.clf.intercept_):
                a1, a2 = coefs
                a2 = a2 * extra_bias
                b = intercept
                graph = ax.plot(lambda x: -(a1 / a2) * x - b / a2, x_range=[-6, 6], use_smoothing=True, color=YELLOW_C)
                l_graph.add(graph)
            iter_graphs.add(l_graph)

        for lr in range(len(self.learning_rate)):
            l_x, l_y, l_z = self.GD(self.learning_rate[lr], start=[2, 2])

            axes = self.ax_3D[0]

            point = Dot3D(axes.c2p(l_x[0], l_y[0], l_z[0]), radius=0.1, color=MAROON)
            graph = iter_graphs[0]
            self.play(Create(point),
                      FadeIn(graph))
            self.wait(2)

            arrow_vg = VGroup()
            for i in range(1, len(l_x)):
                arrow = Arrow(start=axes.c2p(l_x[i - 1], l_y[i - 1], l_z[i - 1]),
                              end=axes.c2p(l_x[i], l_y[i], l_z[i]), buff=0, color=YELLOW_A, stroke_width=8)
                self.play(Write(arrow))
                self.play(point.animate.move_to(axes.c2p(l_x[i], l_y[i], l_z[i])),
                          Transform(graph, iter_graphs[i]))
                arrow_vg.add(arrow)

            self.wait(1)

            self.play(Indicate(point),
                      ApplyWave(graph))

            self.wait(2)

            # self.iterations = VGroup(point, arrow_vg)
            # self.loss_track = [l_x, l_y, l_z]
            # ax = CommonFunc.add_axes(x_range=[0, self.m], y_range=[0, 8], x_length=8, y_length=3,
            #                          axis_config={"include_tip": False, "include_numbers": False}).scale(0.5).to_edge(
            #     RIGHT)
            #
            # labels = ax.get_axis_labels(Text("m").scale(0.5), MathTex("\mathcal{L}").scale(0.45))
            #
            # gd_plot = ax.plot_line_graph(x_values=list(range(self.m)), y_values=self.loss_track[-1],
            #                              line_color=GOLD_E,
            #                              vertex_dot_style=dict(stroke_width=1, fill_color=PURPLE),
            #                              stroke_width=3)
            # # gd_label = ax.get_graph_label(graph=gd_plot, label=MathTex('\\text{lr}=2').scale(0.7), direction=UP)
            #
            # self.loss_graph = VGroup(ax, labels, gd_plot)
            #
            # self.play(Create(self.loss_graph))
            #
            # self.wait(2)
            #
            # self.play(FadeOut(self.iterations))


class llm_train(Scene):
    def construct(self):
        self.text = None
        self.token =None
        self.NN_node = None
        self.NN_weight= None
        self.bert_data = VGroup()

        self.sentence_example()
        self.nn()
        self.BERT_train()
        self.GPT_train()

    def sentence_example(self):
        svg_image = SVGMobject('svg_icon/文档.svg', fill_color=WHITE).scale(0.9).to_edge(LEFT+2*UP)

        self.play(FadeIn(svg_image))
        self.wait(2)

        s_text = '我给你瘦落的街道，绝望的落日，荒郊的月亮。' \
                 '我给你一个久久地望着孤月的人的悲哀。' \
                 '我给你我已死去的祖辈后人们用大理石祭奠的先魂。' \
                 '我父亲的父亲阵亡于布宜诺斯艾利斯的边境,两颗子弹射穿了他的胸膛。' \
                 '死的时候蓄着胡子，尸体被士兵们用牛皮裹起。' \
                 '我母亲的祖父那年才二十四岁，在秘鲁率领三百人冲锋，如今都成了消失的马背上的亡魂。' \
                 '我给你我的书中所能蕴含的一切悟力，以及我生活中所能有的男子气概和幽默。' \
                 '我给你一个从未有过信仰的人的忠诚。' \
                 '我给你我设法保全的我自己的核心。' \
                 '不营字造句，不和梦交易，不被时间、欢乐和逆境触动的核心。' \
                 '我给你早在你出生前多年的一个傍晚看到的一朵黄玫瑰的记忆。' \
                 '我给你关于你生命的诠释，关于你自己的理论，你的真实而惊人的存在。' \
                 '我给你我的寂寞，我的黑暗，我心的饥渴。' \
                 '我试图用困惑、危险、失败来打动你。'

        l_text = s_text.split('。')

        text = VGroup(*[Text(i).scale(0.4) for i in l_text])
        text.arrange_submobjects(DOWN, buff=0.2)

        self.play(Write(text),
                  FadeOut(svg_image))
        self.wait(2)

        self.text = text

        s_token = '我 给 你 瘦落 的 街道 ， 绝望 的 落日 ， 荒郊 的 月亮 。 我 给 你 一个 久久地 望着 孤月 的 人 的 悲哀 。 我 给 你 我 已 死去 的 祖辈 后人们 用 大理石 祭奠 的 先 魂 。 我 父亲 的 父亲 阵亡 于 布宜诺斯艾利斯 的 边境 ， 两 颗 子弹 射 穿 了 他 的 胸膛 。 死 的 时候 蓄 着 胡子 ， 尸体 被 士兵们 用 牛皮 裹 起 。 我 母亲 的 祖父 那年 才 二十四岁 ， 在 秘鲁 率领 三百人 冲锋 ， 如今 都 成 了 消失 的 马 背上 的 亡魂 。 我 给 你 我 的 书 中 所 能 蕴含 的 一切 悟力 ， 以及 我 生活 中 所 能 有 的 男子气概 和 幽默 。 我 给 你 一个 从未 有 过 信仰 的 人 的 忠诚 。 我 给 你 我 设法 保全 的 我 自己 的 核心 。 不 营 字 造句 ， 不 和 梦 交易 ， 不 被 时间 、 欢乐 和 逆境 触动 的 核心 。 我 给 你 早 在 你 出生 前 多年 的 一个 傍晚 看到 的 一朵 黄 玫瑰 的 记忆 。 我 给 你 关于 你 生命 的 诠释 ， 关于 你 自己 的 理论 ， 你 的 真实 而 惊人 的 存在 。 我 给 你 我 的 寂寞 ， 我 的 黑暗 ， 我 心 的 饥渴 。 我 试图 用 困惑 、 危险 、 失败 来 打动 你 。'
        self.token = s_token

    def nn(self):
        def get_nodes(n):
            nodes = VGroup(*[Circle(radius=0.22
                                    , stroke_color=BLUE
                                    , stroke_width=2
                                    , fill_color=GRAY
                                    , fill_opacity=0
                                    ) for _ in range(n)])
            nodes.arrange(RIGHT, buff=0.5)
            return nodes
        node1 = get_nodes(8)
        node2 = get_nodes(8)
        node3 = get_nodes(8)

        vg_nodes = VGroup(node1, node2, node3)

        vg_nodes.arrange(DOWN, buff=0.4).shift(DOWN)

        self.NN_node = vg_nodes

        def create_connections(left_layer_nodes, right_layer_nodes):
            # Create VGroup to hold created connections
            connection_group = VGroup()
            # Iterate through right layer nodes
            for l in range(len(right_layer_nodes)):
                # Iterate through left layer nodes
                for r in range(len(left_layer_nodes)):
                    # Create connection line
                    line = DashedLine(start=right_layer_nodes[l].get_edge_center(UP)
                                , end=left_layer_nodes[r].get_edge_center(DOWN)
                                , color=WHITE,
                                stroke_width=0.9,
                                # , stroke_opacity=0.4
                                )

                    # Add to connection group
                    connection_group.add(line)
            return connection_group

        weight_12 = create_connections(vg_nodes[0], vg_nodes[1])
        weight_23 = create_connections(vg_nodes[1], vg_nodes[2])

        self.NN_weight = VGroup(weight_12, weight_23)

    @staticmethod
    def generate_random_numbers_with_sum(n, total_sum):
        if n < 1:
            raise ValueError("Number of elements must be at least 1.")
        prefix_sum = total_sum
        numbers = []
        for _ in range(n - 1):
            # 这里使用random.uniform生成指定区间的随机数
            random_val = random.uniform(0, prefix_sum)
            numbers.append(random_val)
            prefix_sum -= random_val  # 更新剩余的和

        # 计算最后一个数
        numbers.append(prefix_sum)
        numbers = list(map(lambda x: round(x, 2),numbers))
        return numbers

    def BERT_train(self):
        text = self.text
        tokens = self.token

        l_tokens = tokens.split(' ')
        l_array = l_tokens[:27]
        n_circles = len(l_array)
        vg_texts = VGroup(*[Text(i, color=LOGO_WHITE) for i in l_array])
        vg_texts.arrange_submobjects(RIGHT, buff=0.22).scale(0.5).to_edge(3*DOWN)

        self.bert_data.add(vg_texts)

        plain_rec = VGroup()
        for i in range(n_circles):
            if l_array[i] in ['，', '。']:
                integ = Rectangle(width=0.1, height=0.1, color=MAROON, stroke_width=0.9)
            else:
                len_char = len(l_array[i])
                integ = Rectangle(width=0.66*0.5 * len_char, height=0.75*0.5, color=MAROON, stroke_width=0.9)
            integ.move_to(vg_texts[i].get_center())
            plain_rec.add(integ)

        self.bert_data.add(plain_rec)

        self.play(FadeTransform(text, vg_texts))
        self.wait(2)

        svg_image = SVGMobject('../images/NN.svg', fill_color=WHITE).scale(0.8).next_to(self.NN_node, LEFT)
        self.play(FadeIn(svg_image))
        
        self.play(FadeIn(self.NN_node, target_position= svg_image),
                  FadeIn(self.NN_weight, target_position= svg_image))
        
        self.wait(2)

        self.play(FadeIn(plain_rec))
        self.wait(2)

        # mask操作1,月亮 13

        self.play(plain_rec[13].animate.set_opacity(1))
        self.wait(1)

        mask_brace = Brace(plain_rec[13], DOWN)
        mask_text = Text("掩码（masked）").scale(0.3).next_to(mask_brace, 0.5*DOWN)
        self.play(Create(mask_brace),
                  Create(mask_text))
        self.wait(1)

        self.play(FadeOut(self.bert_data.copy(), target_position=self.NN_node[-1]))

        ## 上下文
        front_brace = Brace(plain_rec[0:13], DOWN, color= GREEN)
        front_text = Text("上文").scale(0.4).next_to(front_brace, 0.5 * DOWN)
        back_brace = Brace(plain_rec[14:], DOWN, color= GREEN)
        back_text = Text("下文").scale(0.4).next_to(back_brace, 0.5 * DOWN)

        vg_front_back = VGroup(front_brace, front_text, back_brace, back_text)
        self.play(FadeIn(vg_front_back))

        self.wait(1)
        self.play(FocusOn(plain_rec[13]))

        ## 输出的可能词语
        output_words = ["月亮", "废墟", "杂草", "枯树", "野兽", "砂石", "孤坟", "荒丘", "残垣", "乌鸦"]
        output_prob = [0.1]*10
        l_output_words_text = VGroup(*[Text(i) for i in output_words])
        l_output_words_text.arrange_submobjects(RIGHT, buff=0.2).scale(0.65).to_edge(UP)

        l_output_prob_float = VGroup(*[DecimalNumber(output_prob[j],
                                                     num_decimal_places=1).next_to(l_output_words_text[j], DOWN) for j in range(len(output_prob))])
        self.play(FadeIn(VGroup(l_output_words_text, l_output_prob_float), target_position=self.NN_node[0]))

        ## 输出的概率分布
        colors_chart = [YELLOW]
        colors_chart.extend([WHITE]*9)
        output_chart = BarChart(
            values=[0.1]*10,
            y_range=[0, 1, 10],
            y_length=4,
            x_length=6,
            bar_colors=colors_chart,
            x_axis_config={"font_size": 50},
        ).scale(0.65).to_edge(UP)
        output_lables = output_chart.get_bar_labels(font_size=22)

        vg_output = VGroup(output_chart, output_lables)
        self.wait(2)
        self.play(FadeTransform(VGroup(l_output_words_text, l_output_prob_float), vg_output))

        self.wait(2)
        self.play(vg_output.animate.shift(1.5*LEFT))
        #
        text_loss = MathTex("\mathcal{L}(",
                            "Y_{pred}",
                            ",",
                            "Y_{true}",
                            ")").to_edge(LEFT+UP)

        self.play(Write(text_loss))
        self.wait(2)

        self.bert_data.add(text_loss)

        self.play(Indicate(vg_output),
                  Indicate(text_loss[1]))
        self.wait(1)

        ## 真实的可能词语
        real_words = ["月亮", "废墟", "杂草", "枯树", "野兽", "砂石", "孤坟", "荒丘", "残垣", "乌鸦"]
        real_prob = [0.99]+[0]*9
        l_real_words_text = VGroup(*[Text(i) for i in real_words])
        l_real_words_text.arrange_submobjects(RIGHT, buff=0.2).scale(0.65).to_edge(2*UP)

        l_real_prob_float = VGroup(*[DecimalNumber(real_prob[j],
                                                     num_decimal_places=2).next_to(l_real_words_text[j], DOWN) for j in range(len(real_prob))])
        self.play(Indicate(text_loss[-2]))
        self.wait(1)
        self.play(FadeTransform(text_loss[-2].copy(), VGroup(l_real_words_text,l_real_prob_float)),
                  FadeOut(vg_output))
        self.wait(1)
        self.play(Indicate(vg_texts[13]),
                  Indicate(VGroup(l_real_words_text,l_real_prob_float)))
        self.wait(1)

        ## 真实的概率分布
        real_chart = BarChart(
            values=[0.99]+ [0]*9,
            y_range=[0, 1, 10],
            y_length=4,
            x_length=6,
            bar_colors=[YELLOW, WHITE],
            x_axis_config={"font_size": 50}).scale(0.65).next_to(output_chart, RIGHT)
        real_labels = real_chart.get_bar_labels(font_size=20)

        vg_real = VGroup(real_labels, real_chart)

        self.play(FadeTransform(VGroup(l_real_words_text,l_real_prob_float), vg_real),
                  FadeIn(vg_output))

        self.wait(2)

        ax = CommonFunc.add_axes(x_range=[0, 6], y_range=[0, 25], x_length=6, y_length=4,
                                 axis_config={"include_tip": False, "include_numbers": False}).scale(0.36).next_to(text_loss, DOWN)

        path = VMobject()
        dot = Dot(ax.c2p(0, 25), radius=DEFAULT_DOT_RADIUS, color=RED)
        path.set_points_as_corners([dot.get_center(), dot.get_center()])

        def update_path(path):
            previous_path = path.copy()
            previous_path.add_points_as_corners([dot.get_center()])
            path.become(previous_path)
        path.add_updater(update_path)

        self.play(FadeIn(ax),
                  FadeTransform(text_loss.copy(), dot))

        self.wait(1)

        self.add(path)
        #
        for x in np.linspace(0.1, 5.1, 10):
            self.play(dot.animate.move_to(ax.c2p(x, (x-5.1)**2)))
            scale_bar = (x-0.1)*0.8/5+0.1
            next_chart = BarChart(
                values=[round(scale_bar, 2)]+self.generate_random_numbers_with_sum(9, 1-scale_bar),
                y_range=[0, 1, 10],
                y_length=4,
                x_length=6,
                bar_colors=[YELLOW]+[WHITE]*9,
                x_axis_config={"font_size": 50},
            ).scale(0.65).move_to(output_chart.get_center())
            next_labels = next_chart.get_bar_labels(font_size=22)

            self.play(Transform(output_chart, next_chart),
                      Transform(output_lables, next_labels),
                      ShowPassingFlash(self.NN_weight.copy().set_color(YELLOW)),
                      ApplyWave(self.NN_node))

        self.wait(2)

        self.play(Indicate(output_chart),
                  Indicate(real_chart))
        self.wait(1)

        # bert
        bert_text = Text('BERT').scale(0.8).to_edge(2*RIGHT)
        self.play(FadeTransform(vg_front_back, bert_text))

        # 多层级的mask

        mask_text.add_updater(lambda m: m.next_to(mask_brace, 0.5*DOWN))

        l_mask_index = [5, 9, 19, 25, 21]
        vg_mask = VGroup()
        for i in l_mask_index:
            self.play(plain_rec.animate.set_opacity(0.1))
            self.play(plain_rec[i].animate.set_opacity(1),
                      mask_brace.animate.next_to(plain_rec[i], DOWN))
            vg_mask.add(plain_rec[i])

        # 按照比例mask
        self.play(plain_rec.animate.set_opacity(0.1),
                  FadeOut(mask_brace),
                  FadeOut(mask_text))
        self.wait(1)
        self.play(vg_mask.animate.set_opacity(1))

        self.wait(2)

        self.play(FadeOut(vg_output),
                  FadeOut(vg_real),
                  FadeOut(VGroup(ax,dot,path)),
                  text_loss.animate.next_to(self.NN_node, 3*UP),
                  FadeOut(bert_text))

        self.wait(1)

        sum_text = MathTex('\sum', color=MAROON).next_to(text_loss, LEFT)
        self.play(FadeTransform(vg_mask, sum_text))
        self.wait(2)

    def GPT_train(self):
        vg_texts = self.bert_data[0]
        plain_rec = self.bert_data[1]
        text_loss = self.bert_data[2]

        ## 从"街道"开始截断
        self.play(FadeOut(vg_texts[7:]))
        front_brace = Brace(plain_rec[0:7], DOWN, color= GREEN)
        front_text = Text("上文").scale(0.4).next_to(front_brace, 0.5 * DOWN)

        vg_front_back = VGroup(front_brace, front_text)
        self.play(FadeIn(vg_front_back))

        # GPT
        GPT_text = Text('GPT').scale(0.8).to_edge(2*RIGHT)
        self.play(FadeTransform(vg_front_back, GPT_text))

        output_chart = BarChart(
            values=self.generate_random_numbers_with_sum(10, 1),
            y_range=[0, 1, 10],
            y_length=4,
            x_length=6,
            bar_colors=[YELLOW]+[WHITE]*9,
            x_axis_config={"font_size": 50},
        ).scale(0.65).to_edge(LEFT+UP)
        output_lables = output_chart.get_bar_labels(font_size=22)

        vg_output = VGroup(output_chart, output_lables)

        real_chart = BarChart(
            values=[0.99] + [0]*9,
            y_range=[0, 1, 10],
            y_length=4,
            x_length=6,
            bar_colors=[YELLOW]+[WHITE]*9,
            x_axis_config={"font_size": 50}).scale(0.65).next_to(text_loss, RIGHT).shift(UP)
        real_lables = real_chart.get_bar_labels(font_size=22)

        vg_real = VGroup(real_chart, real_lables)

        for i in range(7, 27):
            self.play(FadeOut(vg_texts[0:i].copy(), target_position=self.NN_node[-1]))
            self.play(Indicate(text_loss[1]))
            if i == 7:
                self.play(FadeIn(vg_output, target_position=text_loss[1]))
            else:
                new_chart = BarChart(
                    values=self.generate_random_numbers_with_sum(10, 1),
                    y_range=[0, 1, 10],
                    y_length=4,
                    x_length=6,
                    bar_colors=[YELLOW] + [WHITE] * 9,
                    x_axis_config={"font_size": 50},
                ).scale(0.65).to_edge(LEFT + UP)
                new_lables = new_chart.get_bar_labels(font_size=22)
                vg_new = VGroup(new_chart, new_lables)
                self.play(Transform(vg_output, vg_new))
            self.play(FocusOn(plain_rec[i]))
            if i == 7:
                self.play(FadeIn(vg_real, target_position=text_loss[-2]))
            else:
                self.play(Indicate(vg_real),
                          Indicate(text_loss[-2]))
            self.play(FadeIn(vg_texts[i]))

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
              for coord in list(zip(X[:, 0], X[:, 1], kmeans.labels_)) if coord[-1] == 2])
        kmeans_dots_orange = VGroup(
            *[Dot(ax.c2p(coord[0], coord[1]), radius=0.5 * DEFAULT_DOT_RADIUS, color=WHITE)
              for coord in list(zip(X[:, 0], X[:, 1], kmeans.labels_)) if coord[-1] == 0])
        kmeans_dots_green = VGroup(
            *[Dot(ax.c2p(coord[0], coord[1]), radius=0.5 * DEFAULT_DOT_RADIUS, color=WHITE)
              for coord in list(zip(X[:, 0], X[:, 1], kmeans.labels_)) if coord[-1] == 1])
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
class next_chapter(Scene):
    def construct(self):
        title = Text("下一期预告").to_edge(UP)
        self.play(Write(title))

        pass

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
