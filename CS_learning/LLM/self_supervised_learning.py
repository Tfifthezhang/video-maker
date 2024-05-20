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
        self.NN_func = VGroup()

        self.linear_cant()
        self.feature_trans()
        self.represent_nn()

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
        graph = ax.plot(lambda x: x+0.5, x_range=[-0.8, 0.5], use_smoothing=True, color=YELLOW)
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

    def represent_nn(self):
        self.play(FadeOut(self.PCA_example))
        self.wait(2)


# class unsupervised_supervised(Scene):
#     def contruct(self):
#         pass
#     def supervised(self):
#         pass

class supervised_example(ThreeDScene):
    def construct(self):
        self.text = None
        self.ax = VGroup()
        self.m = 10
        self.loss_track = None
        self.loss_graph = None
        self.iterations = None
        self.learning_rate = [0.3]

        self.write_text_before_animate()
        self.get_surfaceplot()
        self.get_dot()

        # self.move_camera(phi=45 * DEGREES, theta=0)
        # l_x, l_y, l_z = self.GD(pow(2, -8) * 20, start=[-2, 2])

    def write_text_before_animate(self):
        text_gradient = MathTex("\mathbf{\omega}_{m+1} =",
                                "\mathbf{\omega}_m - ",
                                "\epsilon","\mathbf{g}(\omega_m)").to_edge(LEFT + UP)
        text_gradient[2].set_color(MAROON)

        g_text = MathTex('\mathbf{g} = \\nabla \mathcal{L}').scale(0.6).next_to(text_gradient, DOWN)

        self.play(Write(text_gradient), Write(g_text))
        self.wait(2)

        self.text = text_gradient

    def get_surfaceplot(self):
        def bowl(u, v):
            z = 1 / 2 * (np.power(u, 2) + np.power(v, 2))
            return z

        axes = ThreeDAxes(x_range=(-4, 4), y_range=(-4, 4), z_range=(0, 7), x_length=8, y_length=8, z_length=4).shift(
            2 * LEFT)
        self.ax.add(axes)
        surface_plane = Surface(lambda u, v: axes.c2p(u, v, bowl(u, v)),
                                u_range=[-2, 2],
                                v_range=[-2, 2],
                                resolution=(30, 30),
                                should_make_jagged=True,
                                stroke_width=0.2)

        surface_plane.set_style(fill_opacity=0.5, stroke_color=RED)

        surface_plane.set_fill_by_value(axes=axes, colors=[(RED, 0.0), (YELLOW, 0.2), (BLUE, 4)], axis=2)

        self.ax.add(surface_plane)

        self.play(Create(self.ax))

        self.play(Rotate(self.ax, angle=-90 * DEGREES, axis=RIGHT))
        self.wait(2)

        # self.move_camera(phi=75 * DEGREES)

        # self.set_camera_orientation(phi=75 * DEGREES, theta=0)

        for i in range(6):
            self.play(Rotate(self.ax, angle=60 * DEGREES, axis=UP))

        self.play(Rotate(self.ax, angle=15 * DEGREES, axis=UP))
        self.play(Rotate(self.ax, angle=45 * DEGREES, axis=RIGHT))

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
        for lr in range(len(self.learning_rate)):
            l_x, l_y, l_z = self.GD(self.learning_rate[lr], start=[2, 2])

            axes = self.ax[0]

            point = Dot3D(axes.c2p(l_x[0], l_y[0], l_z[0]), radius=0.1, color=MAROON)
            self.play(Create(point))
            self.wait(2)

            arrow_vg = VGroup()
            for i in range(1, len(l_x)):
                arrow = Arrow(start=axes.c2p(l_x[i - 1], l_y[i - 1], l_z[i - 1]),
                              end=axes.c2p(l_x[i], l_y[i], l_z[i]), color=GRAY, stroke_width=5)
                self.play(Write(arrow))
                self.play(point.animate.move_to(axes.c2p(l_x[i], l_y[i], l_z[i])))

                arrow_vg.add(arrow)

            self.wait(2)

            self.iterations = VGroup(point, arrow_vg)
            self.loss_track = [l_x, l_y, l_z]
            ax = CommonFunc.add_axes(x_range=[0, self.m], y_range=[0, 8], x_length=8, y_length=3,
                                     axis_config={"include_tip": False, "include_numbers": False}).scale(0.5).to_edge(RIGHT)

            labels = ax.get_axis_labels(Text("m").scale(0.5), MathTex("\mathcal{L}").scale(0.45))

            gd_plot = ax.plot_line_graph(x_values=list(range(self.m)), y_values=self.loss_track[-1],
                                         line_color=GOLD_E,
                                         vertex_dot_style=dict(stroke_width=1, fill_color=PURPLE),
                                         stroke_width=3)
            # gd_label = ax.get_graph_label(graph=gd_plot, label=MathTex('\\text{lr}=2').scale(0.7), direction=UP)

            self.loss_graph = VGroup(ax, labels, gd_plot)

            self.play(Create(self.loss_graph))

            self.wait(2)

            self.play(FadeOut(self.iterations))

class llm_train(Scene):
    def construct(self):
        self.example_texts = None

        self.sentence_example()

    def sentence_example(self):
        l_array = list('迷路的小画家是个英俊少年')
        n_circles = len(l_array)

        circles = VGroup(*[Circle(radius=0.4,color=LOGO_WHITE).scale(1)
                           for _ in range(n_circles)])
        circles.arrange_submobjects(RIGHT, buff=0.15)

        texts = VGroup()
        for i in range(n_circles):
            integ = Text(l_array[i], color=LOGO_WHITE).scale(0.8)
            integ.move_to(circles[i].get_center())
            texts.add(integ)

        circle_texts = VGroup(circles, texts)

        self.play(Create(circle_texts))

        self.wait(2)

        self.example_texts = circle_texts

    def BERT_train(self):
        pass



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
