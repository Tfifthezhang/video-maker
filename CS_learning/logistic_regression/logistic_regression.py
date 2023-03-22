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

sys.path.append('..')

from CS_learning.common_func import CommonFunc
import scipy.stats as stats

from sklearn.datasets import make_moons, make_blobs, make_classification

np.random.seed(0)


class Classification(MovingCameraScene):
    def construct(self):
        self.linear_ax = VGroup()
        self.class_ax = VGroup()
        self.sigmoid_ax = VGroup()
        self.formula = VGroup()

        self.linear_regression()
        self.Write_Formula()
        self.linear_trans_classification()
        self.make_01_classification()
        self.Formula_trans()
        self.sigmoid_plot()
        self.function_shift()

    def Write_Formula(self):
        weight_formula = MathTex('y = \omega x + b').to_edge(UP + LEFT)
        self.play(Write(weight_formula))
        self.wait(2)

        self.formula.add(weight_formula)

    def linear_regression(self):
        ax = CommonFunc.add_axes(x_range=[-8, 8], y_range=[-8, 8], x_length=8, y_length=6,
                                 axis_config={"include_tip": False, "include_numbers": False})
        self.play(Create(ax))
        self.linear_ax.add(ax)

        axes_labels = ax.get_axis_labels(x_label=MathTex('x'), y_label=MathTex('y'))
        self.play(Create(axes_labels))
        self.linear_ax.add(axes_labels)

        x = np.linspace(-7.5, 7.5, 150)
        gaussian_noise = np.random.normal(loc=0, scale=3, size=(150,))
        y = 0.8 * x - 0.7
        y_noise = y + gaussian_noise
        coords = list(zip(x, y_noise))

        dots = VGroup(
            *[Dot(ax.c2p(coord[0], coord[1]), radius=0.5 * DEFAULT_DOT_RADIUS, color=BLUE) for coord in coords])
        self.linear_ax.add(dots)
        self.play(FadeIn(dots))

        inital_plot = ax.plot(lambda x: 0.8 * x - 0.7, x_range=[-7, 7], use_smoothing=True, color=MAROON)
        inital_label = ax.get_graph_label(graph=inital_plot, label=MathTex('y = 0.8x - 0.7')).scale(0.8)

        self.play(Write(inital_plot), Write(inital_label))
        self.linear_ax.add(inital_label)
        self.linear_ax.add(inital_plot)

        regression_brace = Brace(dots, LEFT)
        regression_text = MathTex('y \in \mathbb{R}', color=MAROON).next_to(regression_brace, LEFT)

        self.play(Create(regression_brace), Write(regression_text))
        self.linear_ax.add(regression_brace)
        self.linear_ax.add(regression_text)

        self.wait(2)

    def linear_trans_classification(self):
        n, p = 1, .5
        s = np.random.binomial(n, p, 150)
        x = np.linspace(-7.5, 7.5, 150)
        coords = list(zip(x, s))

        dots = VGroup(
            *[Dot(self.linear_ax[0].c2p(coord[0], coord[1]), radius=0.5 * DEFAULT_DOT_RADIUS, color=BLUE) if coord[
                                                                                                                 1] == 0 else
              Dot(self.linear_ax[0].c2p(coord[0], coord[1]), radius=0.5 * DEFAULT_DOT_RADIUS, color=RED) for coord in
              coords])

        class_brace = Brace(dots, LEFT)
        class_text = MathTex('y \in \{0,1\}', color=MAROON).next_to(self.linear_ax[-2], LEFT)

        self.play(FadeOut(self.linear_ax[3]),
                  FadeOut(self.linear_ax[4]),
                  Transform(self.linear_ax[2], dots),
                  Transform(self.linear_ax[-2], class_brace),
                  Transform(self.linear_ax[-1], class_text))
        self.wait(3)

    def make_01_classification(self):
        ax = CommonFunc.add_axes(x_range=[-3, 3], y_range=[-3, 3], x_length=8, y_length=8,
                                 axis_config={"include_tip": False, "include_numbers": False})

        axes_labels = ax.get_axis_labels(x_label=MathTex('x_1'), y_label=MathTex('x_2'))

        self.class_ax.add(ax)
        self.class_ax.add(axes_labels)

        centers = [[-1, 1], [1, -1]]

        X, y = make_blobs(
            n_samples=300, centers=centers, cluster_std=0.45, random_state=0
        )
        coords = list(zip(X, y))

        dots = VGroup(
            *[Dot(ax.c2p(coord[0][0], coord[0][1]), radius=0.5 * DEFAULT_DOT_RADIUS, color=BLUE) if coord[1] == 0 else
              Dot(ax.c2p(coord[0][0], coord[0][1]), radius=0.5 * DEFAULT_DOT_RADIUS, color=RED) for coord in
              coords])

        self.class_ax.add(dots)

        # ax_cn_text = Text('特征空间').scale(0.6).next_to(ax, UP + RIGHT)
        # ax_en_text = Text('attribute space').scale(0.5).next_to(ax_cn_text, DOWN)
        #
        # self.class_ax.add(VGroup(ax_cn_text, ax_en_text))

        self.play(ReplacementTransform(self.linear_ax, self.class_ax))

        self.wait(3)

    def Formula_trans(self):
        weight_formula = MathTex('\eta = \omega x + b').to_edge(UP + LEFT)

        self.play(Transform(self.formula, weight_formula))

        g_formula = MathTex("y", "=", "g(\eta)").next_to(weight_formula, DOWN)

        self.play(Write(g_formula))
        self.formula.add(g_formula)
        self.wait(1)

        function_formula = MathTex('\mathbb{R} \\rightarrow \{0, 1\}', color=MAROON).scale(0.8).next_to(g_formula, DOWN)
        self.play(FadeTransform(g_formula[-1].copy(), function_formula))
        self.formula.add(function_formula)

        self.wait(3)

    def sigmoid_plot(self):
        self.play(self.class_ax.animate.scale(0.75).to_edge(RIGHT))
        self.wait(2)

        ax = CommonFunc.add_axes(x_range=[-8, 8], y_range=[0, 1], x_length=7, y_length=4,
                                 axis_config={"include_tip": False, "include_numbers": True})

        axes_labels = ax.get_axis_labels(x_label=MathTex('\eta'), y_label=MathTex('y'))

        self.sigmoid_ax.add(ax)
        self.sigmoid_ax.add(axes_labels)

        inital_plot = ax.plot(lambda x: self.H(x), x_range=[-7, 7], use_smoothing=False, color=MAROON)
        inital_label = ax.get_graph_label(graph=inital_plot, label=MathTex('y=H(\eta)').scale(0.6), direction=DOWN)

        self.sigmoid_ax.add(inital_plot)
        self.sigmoid_ax.add(inital_label)

        self.sigmoid_ax.scale(0.8).next_to(self.class_ax, LEFT)

        self.play(FadeTransform(self.formula[-1], self.sigmoid_ax))

        self.wait(3)

        sigmoid_plot = ax.plot(lambda x: self.sigmoid(x), x_range=[-7, 7], use_smoothing=True, color=MAROON)
        sigmoid_label = ax.get_graph_label(graph=inital_plot, label=MathTex('y=\\frac{1}{1+e^{-\eta}}').scale(0.6),
                                           direction=DOWN)

        self.play(Transform(inital_plot, sigmoid_plot),
                  Transform(inital_label, sigmoid_label))
        self.wait(3)

    def sigmoid(self, x):
        s = np.power(np.e, -x)
        return 1 / (1 + s)

    def H(self, x):
        if x == 0:
            return 0.5
        if x > 0:
            return 1
        else:
            return 0

    def function_shift(self):
        zero_p = self.sigmoid_ax[0].c2p(0, 0.5)
        zero_dot = Dot(zero_p)
        # line = self.sigmoid_ax[0].get_horizontal_line(zero_p, line_func=Line)
        zero_des = MathTex("(0,0.5)").scale(0.6).next_to(zero_dot, 0.2 * RIGHT)

        self.play(Write(zero_dot), Write(zero_des))

        self.wait(3)

        left_arrow = CommonFunc.add_arrow(self.sigmoid_ax[0].c2p(-3.5, 0.5),
                                          self.sigmoid_ax[0].c2p(-6, 0.5), color=GREEN)

        right_arrow = CommonFunc.add_arrow(self.sigmoid_ax[0].c2p(3.5, 0.5),
                                           self.sigmoid_ax[0].c2p(6, 0.5), color=GREEN)

        self.play(Write(left_arrow))
        self.wait(1)
        self.play(Write(right_arrow))

        self.wait(3)

        # 将决策边界映射到特征空间
        self.play(self.class_ax[1].animate.set_style(fill_opacity=0.3))

        inital_plot = self.class_ax[0].plot(lambda x: x, x_range=[-2.5, 2.5], use_smoothing=True, color=MAROON)
        inital_label = self.class_ax[0].get_graph_label(graph=inital_plot,
                                                        label=MathTex('\omega_1 x_1+\omega_2 x_2+b=0').scale(0.55),
                                                        direction=LEFT + DOWN)

        self.play(Indicate(VGroup(zero_dot, zero_des, self.formula[0])))

        self.wait(1)

        self.play(FadeTransform(VGroup(zero_dot.copy(),
                                       zero_des.copy(),
                                       self.formula[0].copy()),
                                VGroup(inital_plot,
                                       inital_label), run_time=1))
        self.wait(3)

        up_arrow = CommonFunc.add_arrow(self.class_ax[0].c2p(0.5, 1),
                                        self.class_ax[0].c2p(0.5, 2), color=GREEN)

        down_arrow = CommonFunc.add_arrow(self.class_ax[0].c2p(-0.5, -1),
                                          self.class_ax[0].c2p(-0.5, -2), color=GREEN)

        self.play(FadeTransform(right_arrow.copy(), up_arrow, run_time=1))
        self.play(FadeTransform(left_arrow.copy(), down_arrow, run_time=1))

        self.wait(3)

        self.play(FadeOut(up_arrow), FadeOut(down_arrow))

        # 决策边界的作用示意图
        self.camera.frame.save_state()

        self.play(self.camera.frame.animate.scale(0.8).move_to(self.class_ax))

        l_omega_2 = np.linspace(1.4, 0.5, 20)
        l_omega_1 = -np.linspace(0.5, 1.4, 20) * l_omega_2
        l_b = -np.linspace(1.5, -1.5, 20) * l_omega_2
        for omega_2, omega_1, b in zip(l_omega_2, l_omega_1, l_b):
            next_plot = self.class_ax[0].plot(lambda x: -omega_1 / omega_2 * x - b / omega_2, x_range=[-2.5, 2.5],
                                              use_smoothing=True, color=MAROON)
            next_label = self.class_ax[0].get_graph_label(graph=next_plot,
                                                          label=MathTex(
                                                              '{0:.2f}x_1+{1:.2f}x_2+{2:.2f}=0'.format(omega_1, omega_2,
                                                                                                       b))).scale(0.55)
            self.play(Transform(inital_plot, next_plot), Transform(inital_label, next_label))
            blue_vg = VGroup()
            red_vg = VGroup()
            for dot in self.class_ax[2]:
                x, y = self.class_ax[0].p2c(dot.get_center())
                if omega_1 * x + omega_2 * y + b > 0:
                    blue_vg.add(dot)
                else:
                    red_vg.add(dot)
            self.play(blue_vg.animate.set_color(BLUE, family=True))
            self.play(red_vg.animate.set_color(RED, family=True))


class logistic_3D(ThreeDScene):
    def construct(self):
        self.class_ax = VGroup()

        # self.set_camera_orientation(zoom=1, frame_center=ORIGIN)
        # self.Write_formula()
        # self.make_01_classification()
        # self.sigmoid_3D()
        # self.sigmoid_bernoulli()

    def Write_formula(self):
        formula = MathTex('P(y=1|\eta) =\\frac{1}{1+e^{-\eta}}').scale(0.8).to_edge(UP + LEFT)
        self.play(Write(formula))

    def make_01_classification(self):
        axes = ThreeDAxes(x_range=(-6, 6), y_range=(-6, 6), z_range=(-0.15, 1.15), x_length=7, y_length=7, z_length=4)
        # z_label = axes.get_z_axis_label(MathTex('S(x_1,x_2)'))

        # axes_labels = axes.get_axis_labels(x_label=MathTex('x_1'), y_label=MathTex('x_2'))

        self.class_ax.add(axes)

        centers = [[-1, 1], [1, -1]]

        X, y = make_blobs(
            n_samples=100, centers=centers, cluster_std=0.45, random_state=0
        )
        coords = list(zip(X, y))

        dots = VGroup(
            *[Dot3D(axes.c2p(coord[0][0], coord[0][1], 0), radius=0.5 * DEFAULT_DOT_RADIUS, color=BLUE) if coord[
                                                                                                               1] == 0 else
              Dot3D(axes.c2p(coord[0][0], coord[0][1], 1), radius=0.5 * DEFAULT_DOT_RADIUS, color=RED) for coord in
              coords])

        self.class_ax.add(dots)

        self.play(Create(self.class_ax))
        # self.move_camera(phi=75 * DEGREES)
        self.wait(3)

        self.play(Rotate(self.class_ax, angle=-90 * DEGREES, axis=RIGHT))

        # for i in range(6):
        #     self.play(Rotate(self.class_ax, angle=60 * DEGREES, axis=UP))

        self.wait(3)

    def sigmoid_3D(self):

        surface_1 = Surface(lambda u, v: self.class_ax[0].c2p(u, v, self.func_sigmoid(u, v)),
                            u_range=[-5, 5],
                            v_range=[-5, 5],
                            resolution=(30, 30),
                            should_make_jagged=True,
                            stroke_width=0.2,
                            )

        surface_1.set_style(fill_opacity=0.3, stroke_color=RED)

        surface_1.set_fill_by_value(axes=self.class_ax[0], colors=[(BLUE, 0.5), (RED, 1)], axis=2)
        #
        # surface_2 = Surface(lambda u, v: self.class_ax[0].c2p(u, v, self.func_sigmoid(-u, -v)),
        #                     u_range=[-5, 5],
        #                     v_range=[-5, 5],
        #                     resolution=(40, 40),
        #                     should_make_jagged=True,
        #                     stroke_width=0.2,
        #                     )
        #
        # surface_2.set_style(fill_opacity=0.5, stroke_color=RED)
        #
        # surface_2.set_fill_by_value(axes=self.class_ax[0], colors=[(BLUE, 0.25), (YELLOW, 0.75), (RED, 1)], axis=2)

        self.class_ax.add(surface_1)
        # self.class_ax.add(surface_2)

        self.play(Create(surface_1))

        self.wait(3)

        for i in range(6):
            self.play(Rotate(self.class_ax, angle=60 * DEGREES, axis=UP))

        # self.play(Rotate(self.class_ax, angle=90 * DEGREES, axis=RIGHT))

    def sigmoid_bernoulli(self):
        dot = self.class_ax[1][0]
        # self.camera.save_state()

        self.move_camera(frame_center=dot, zoom=2)
        # self.play(self.camera.frame.animate.scale(0.5).move_to(dot))

        inital_chart = BarChart(
            values=[0.5, 0.5],
            bar_names=["0", "1"],
            y_range=[0, 1, 10],
            y_length=4,
            x_length=2,
            x_axis_config={"font_size": 36},
        ).scale(0.3).next_to(dot, 0.1 * UP)

        self.play(Create(inital_chart))  # Create(c_bar_lbls))

        bernoulli_formula = MathTex('\mathcal{B}(P)').scale(0.4).next_to(inital_chart, RIGHT)
        self.play(FadeIn(bernoulli_formula))
        self.wait(3)

        l_p = np.random.random(10)
        for p in l_p:
            dist = stats.bernoulli(p)
            value = [dist.pmf(0), dist.pmf(1)]
            chart = BarChart(
                values=list(map(lambda x: round(x, 2), value)),
                bar_names=["0", "1"],
                y_range=[0, 1, 10],
                y_length=4,
                x_length=2,
                x_axis_config={"font_size": 36},
            ).scale(0.4).next_to(dot, 0.1 * UP)

            self.play(Transform(inital_chart, chart))

        self.wait(2)

        bernoulli_p = MathTex('P(y=1|\eta) =\\frac{1}{1+e^{-\eta}}', color=MAROON).scale(0.4).next_to(bernoulli_formula,
                                                                                                      RIGHT)

        self.play(Write(bernoulli_p))

        self.wait(1)

        bernoulli_final = MathTex('\mathcal{B}(\\frac{1}{1+e^{-\eta})').scale(0.4).next_to(inital_chart, RIGHT)

        self.play(FadeTransform(VGroup(bernoulli_formula, bernoulli_p), bernoulli_final))

        self.wait(2)

        self.move_camera(frame_center=ORIGIN, zoom=1)

        other_dot = self.class_ax[1][-1]
        dist = stats.bernoulli(np.random.rand())
        value = [dist.pmf(0), dist.pmf(1)]
        chart = BarChart(
            values=list(map(lambda x: round(x, 2), value)),
            bar_names=["0", "1"],
            y_range=[0, 1, 10],
            y_length=4,
            x_length=2,
            x_axis_config={"font_size": 36},
        ).scale(0.4).next_to(other_dot, 0.1 * UP)
        self.play(FadeIn(chart))

        self.play(Rotate(self.class_ax, angle=90 * DEGREES, axis=RIGHT))

    def func_sigmoid(self, x_1, x_2):
        s = np.power(np.e, -(x_1 - x_2))
        return 1 / (1 + s)


class InSVG(Scene):
    def construct(self):
        self.insert_coutour()

    def insert_coutour(self):
        svg_coutour = SVGMobject('logistic_regression/4.svg', height=7, width=7)
        self.play(Create(svg_coutour))
        for i in range(6):
            self.play(Rotate(svg_coutour, angle=60 * DEGREES, axis=OUT))


class Other_01Function(Scene):
    def construct(self):
        self.sigmoid_ax = VGroup()
        self.formula = VGroup()

        self.Formula_trans()
        self.sigmoid_function()
        self.other_function()
        # self.exponential_family()
        # self.general_form()

    def Formula_trans(self):
        weight_formula = MathTex('\eta = \omega x + b').to_edge(UP + LEFT)

        self.add(weight_formula)
        self.formula.add(weight_formula)

        g_formula = MathTex("y", "=", "g(\eta)").next_to(weight_formula, DOWN)

        self.add(g_formula)
        self.formula.add(g_formula)

        function_formula = MathTex('\mathbb{R} \\rightarrow \{0, 1\}', color=MAROON).scale(0.8).next_to(g_formula, DOWN)
        self.add(function_formula)
        self.formula.add(function_formula)

        self.wait(2)

    def sigmoid_function(self):
        ax = CommonFunc.add_axes(x_range=[-8, 8], y_range=[0, 1], x_length=7, y_length=4,
                                 axis_config={"include_tip": False, "include_numbers": True})

        axes_labels = ax.get_axis_labels(x_label=MathTex('\eta'), y_label=MathTex('y'))

        self.sigmoid_ax.add(ax)
        self.sigmoid_ax.add(axes_labels)

        inital_plot = ax.plot(lambda x: self.H(x), x_range=[-7, 7], use_smoothing=False, color=MAROON)
        inital_label = ax.get_graph_label(graph=inital_plot, label=MathTex('y=H(\eta)').scale(0.6), direction=DOWN)

        self.sigmoid_ax.add(inital_plot)
        self.sigmoid_ax.add(inital_label)

        self.play(FadeIn(self.sigmoid_ax))

        sigmoid_plot = ax.plot(lambda x: self.sigmoid(x), x_range=[-7, 7], use_smoothing=True, color=MAROON)
        sigmoid_label = ax.get_graph_label(graph=sigmoid_plot, label=MathTex('y=\\frac{1}{1+e^{-\eta}}').scale(0.6),
                                           direction=DOWN)

        self.play(Transform(inital_plot, sigmoid_plot),
                  Transform(inital_label, sigmoid_label))

        self.wait(3)

        self.play(self.sigmoid_ax.animate.scale(0.6).to_edge(LEFT))

    def other_function(self):
        vg_graph = VGroup()
        l_graph_label = [MathTex('arctan(x)'),
                         MathTex('tanh(x)'),
                         MathTex('gd(x)'),
                         MathTex('\\frac{x}{\sqrt{1+{x^2}}}')]
        l_function = [self.arctan, self.tanh, self.goodman, self.daishu]
        for i, j in list(zip(l_graph_label, l_function)):
            ax = CommonFunc.add_axes(x_range=[-8, 8], y_range=[-2, 2], x_length=6, y_length=4,
                                     axis_config={"include_tip": True, "include_numbers": False}).scale(0.8)
            graph = ax.plot(lambda x: j(x), x_range=[-7, 7], color=RED, use_smoothing=True)
            graph_label = ax.get_graph_label(graph=graph,
                                             label=i.scale(1),
                                             color=RED,
                                             direction=UP)

            ax_vg = VGroup(ax, graph, graph_label)
            vg_graph.add(ax_vg)

        vg_graph.arrange_in_grid(2, 2, buff=2 * SMALL_BUFF).scale(0.7).to_edge(RIGHT)
        for graph in vg_graph:
            self.play(FadeTransform(self.formula[-1].copy(), graph))

        self.wait(3)

        self.play(Indicate(self.formula[-1]))

        function_formula_2 = MathTex('\mathbb{R} \\rightarrow \{-1, 1\}', color=GREEN).scale(0.8).next_to(
            self.formula[1], DOWN)

        self.play(Transform(self.formula[-1], function_formula_2, run_time=1))

        self.wait(2)

        # self.play(FadeOut(vg_graph), FadeOut(self.sigmoid_ax), FadeOut(self.formula[-1]))

        self.wait(1)

    def arctan(self, x):
        return np.arctan(x)

    def daishu(self, x):
        return x / np.sqrt(1 + x ** 2)

    def tanh(self, x):
        return np.tanh(x)

    def goodman(self, x):
        return 2 * np.arctan(np.tanh(x / 2))

    def sigmoid(self, x):
        s = np.power(np.e, -x)
        return 1 / (1 + s)

    def H(self, x):
        if x == 0:
            return 0.5
        if x > 0:
            return 1
        else:
            return 0


class GLM_theory(Scene):
    def construct(self):
        self.formula = VGroup()
        self.generalized_formula = VGroup()
        self.exponential_group = None
        self.normal_link_function = None

        self.Formula_trans()
        self.generalized_text()
        self.general_explain()
        self.GLM_intro()
        self.exponential_family()
        self.general_form()

        # self.general_form()

    def Formula_trans(self):
        weight_formula = MathTex('\eta = \omega x + b').to_edge(UP + LEFT)

        self.add(weight_formula)
        self.formula.add(weight_formula)

        g_formula = MathTex("y", "=", "g^{-1}", "(\eta)").next_to(weight_formula, DOWN)

        self.add(g_formula)
        self.formula.add(g_formula)

    def generalized_text(self):
        cn_title = Text('广义线性模型').scale(0.7).to_edge(UP)
        en_title = Text('generalized linear model', color=MAROON).scale(0.5).next_to(cn_title, 0.1 * DOWN)
        self.generalized_formula.add(cn_title)
        self.generalized_formula.add(en_title)

        # self.play(Write(cn_title), Write(en_title))

        assume_1_padding = Text('指数族分布 Exponential family ').scale(0.5).to_edge(3 * UP + RIGHT)
        assume_1_tex = MathTex("1.", "P(y|\eta)", "\sim").scale(0.6).next_to(assume_1_padding, LEFT)
        self.generalized_formula.add(VGroup(assume_1_tex, assume_1_padding))

        # self.play(Circumscribe(self.generalized_formula[2][1]))

        assume_2 = MathTex('2. \eta = \omega x + b').scale(0.65).next_to(assume_1_tex, DOWN).align_to(assume_1_tex[0],
                                                                                                      LEFT)

        assume_3 = MathTex("3.", "\mu = \mathbb{E}(y)", "= g^{-1}(\eta)").scale(0.6).next_to(assume_2, DOWN).align_to(
            assume_2, LEFT)

        self.generalized_formula.add(VGroup(assume_2))
        self.generalized_formula.add(assume_3)

    def general_explain(self):
        model_ax = CommonFunc.add_axes(x_range=[-8, 8], y_range=[-8, 8], x_length=8, y_length=6,
                                       axis_config={"include_tip": False, "include_numbers": False}).scale(0.7)
        model_plot = model_ax.plot(lambda x: 0.8 * x + 0.7, x_range=[-6, 6], use_smoothing=True, color=BLUE)
        model_label = model_ax.get_graph_label(graph=model_plot, label=MathTex('\{\omega, b\,....\}'),
                                               direction=DOWN).scale(0.8)

        model_text = Text('线性模型 Model').scale(0.7).next_to(model_ax, DOWN)

        vg_model = VGroup(model_ax, model_plot, model_label, model_text).scale(0.6).shift(3 * LEFT + DOWN)

        dis_ax = CommonFunc.add_axes(x_range=[-6, 6], y_range=[0, 1], x_length=8, y_length=6,
                                     axis_config={"include_tip": True, "include_numbers": False}).scale(0.7)
        dis_plot = dis_ax.plot(lambda x: self.normal_dis(x, mu=1, sigma=0.8), x_range=[1 - 3, 1 + 3],
                               use_smoothing=True, color=RED)
        dis_label = dis_ax.get_graph_label(graph=dis_plot, label=MathTex('\{\mu,\sigma^2\,....\}'), direction=UP).scale(
            0.8)

        dis_text = Text('假设分布 Distribution').scale(0.7).next_to(dis_ax, DOWN)

        vg_dis = VGroup(dis_ax, dis_plot, dis_label, dis_text).scale(0.6).shift(3 * RIGHT + DOWN)

        space_vg = VGroup(vg_model, vg_dis).arrange_submobjects(RIGHT, buff=3.5).shift(DOWN)

        arrow_1 = CommonFunc.add_arrow(vg_model.get_right(), vg_dis.get_left(), color=WHITE)
        arrow_2 = CommonFunc.add_arrow(vg_dis.get_left(), vg_model.get_right(), color=WHITE).next_to(arrow_1, DOWN)

        vg_arrow = VGroup(arrow_1, arrow_2)

        self.play(FadeTransform(VGroup(self.formula[0], self.formula[1][3]), vg_model))
        self.wait(1)

        self.play(FadeTransform(self.formula[1][:2], vg_dis))
        self.wait(1)

        self.play(FadeTransform(self.formula[1][2], vg_arrow))
        self.wait(1)

        g_1 = MathTex('g^{-1}').scale(0.6).next_to(arrow_1, 0.5 * UP)
        g_2 = MathTex('g').scale(0.6).next_to(arrow_2, 0.5 * DOWN)
        self.play(Write(g_1), Write(g_2))

        vg_arrow.add(g_1)
        vg_arrow.add(g_2)

        self.wait(2)

        # 这三个条件分别对应着广义线性模型的三个假设
        ## 假设分布
        self.play(FadeTransform(vg_dis.copy(), self.generalized_formula[2]))
        self.wait(3)
        ## 线性组合
        self.play(FadeTransform(vg_model.copy(), self.generalized_formula[3]))
        self.wait(3)
        ## 链接函数,分布的均值等于链接函数的反函数
        self.play(FadeTransform(vg_arrow.copy(), self.generalized_formula[4]))
        self.wait(3)

        self.play(FadeIn(self.generalized_formula[0]),
                  FadeIn(self.generalized_formula[1]))

        self.wait(1)

        self.play(Wiggle(self.generalized_formula[4][1], run_time=1))

        self.wait(3)

        self.play(FadeOut(VGroup(vg_dis, vg_model, vg_arrow)))

    def GLM_intro(self):

        # 让线性回归作为一个例子
        previous_example = ImageMobject('logistic_regression/MLE_Regression.png').scale(0.3).to_edge(0.3 * LEFT).shift(
            2 * UP)
        self.add(previous_example)
        self.wait(3)

        example_assume_1 = MathTex("P(y|\eta)", "\sim", "\mathcal{N}(\mu, \sigma^2)").scale(0.65).next_to(
            previous_example, DOWN)

        example_assume_2 = MathTex('\eta = \omega x').scale(0.65).next_to(example_assume_1, DOWN).align_to(
            example_assume_1[0], LEFT)

        example_assume_3 = MathTex('\mu = g^{-1}(\eta) = \eta = \omega x').scale(0.65).next_to(example_assume_2,
                                                                                               DOWN).align_to(
            example_assume_2, LEFT)

        self.play(FadeTransform(self.generalized_formula[2].copy(), example_assume_1))
        self.wait(1)
        self.play(FadeTransform(self.generalized_formula[3].copy(), example_assume_2))
        self.wait(1)
        self.play(FadeTransform(self.generalized_formula[4].copy(), example_assume_3))
        self.wait(3)

        g_tex = MathTex('g = \operatorname{Id}', color=MAROON).scale(0.8).next_to(previous_example, RIGHT)
        self.play(ReplacementTransform(example_assume_3, g_tex))

        self.play(FadeOut(VGroup(example_assume_1, example_assume_2)))
        self.wait(2)

        self.normal_graph = previous_example
        self.normal_link_function = g_tex

    def exponential_family(self):

        def normal(x):
            normal = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)
            return normal

        def laplace(x):
            laplace = 0.5 * np.exp(-np.abs(x))
            return laplace

        def f_gamma(x):
            if x < 0:
                return 0
            gamma = x * np.exp(-x)
            return gamma

        def f_exp(x):
            if x < 0:
                return 0
            exponential = 0.5 * np.exp(-0.5 * x)
            return exponential

        self.play(Circumscribe(self.generalized_formula[2][1]))

        vg_graph = VGroup()
        l_graph_label = [MathTex('\mathcal{N}(x)'),
                         MathTex('Laplace(x)'),
                         MathTex('\Gamma(x)'),
                         MathTex('\exp(x)')]
        l_function = [normal, laplace, f_gamma, f_exp]
        for i, j in list(zip(l_graph_label, l_function)):
            ax = CommonFunc.add_axes(x_range=[-8, 8], y_range=[0, 1], x_length=6, y_length=4,
                                     axis_config={"include_tip": True, "include_numbers": False}).scale(0.7)
            graph = ax.plot(lambda x: j(x), x_range=[-7, 7], color=WHITE, use_smoothing=True)
            graph_label = ax.get_graph_label(graph=graph,
                                             label=i.scale(1),
                                             color=WHITE,
                                             direction=UP)

            ax_vg = VGroup(ax, graph, graph_label)
            vg_graph.add(ax_vg)

        bernoulli_chart = BarChart(
            values=[0.75, 0.25],
            bar_names=["0", "1"],
            y_range=[0, 1, 10],
            y_length=4,
            x_length=6,
            x_axis_config={"font_size": 36},
            bar_colors=[WHITE, WHITE],
        ).scale(0.7)

        bernoulli_label = MathTex('\mathcal{B}(x)').next_to(bernoulli_chart, 0.001 * UP)

        vg_graph.add(VGroup(bernoulli_chart, bernoulli_label))

        vg_graph.arrange_submobjects(RIGHT, buff=0.5 * SMALL_BUFF).scale(0.55).shift(DOWN)

        for graph in vg_graph:
            self.play(FadeTransform(self.generalized_formula[2][1].copy(), graph))

        self.exponential_group = vg_graph

        self.wait(3)

    def general_form(self):

        tex_form = MathTex("P(y \mid \eta)", "=", "h(y)", "\exp", "(", "\eta^T", "T(y)", "-", "A(\eta)", ")").scale(0.8)

        self.play(FadeTransform(self.exponential_group, tex_form))

        self.wait(2)

        # 自然参数：
        self.play(tex_form[5].animate.set_color(GREEN))
        self.wait(3)
        # 充分统计量
        self.play(tex_form[6].animate.set_color(BLUE))
        self.wait(3)
        # 配分函数
        self.play(tex_form[8].animate.set_color(YELLOW))
        self.wait(3)

        normal_form = MathTex('P(y \mid \mu,\sigma)='
                              '\\frac{1}{\sigma \sqrt{2 \pi}} '
                              '\exp (-\\frac{1}{2 \sigma^2}(y-\mu)^2)').scale(0.8).next_to(tex_form, DOWN).align_to(
            tex_form, LEFT)

        self.play(FadeIn(normal_form))
        self.wait(2)

        normal_form_dis = MathTex('P(y \mid \mu)='
                                  '\\frac{1}{\sqrt{2 \pi}} '
                                  '\exp (-\\frac{1}{2}(y-\mu)^2)').scale(0.8).next_to(tex_form, DOWN).align_to(
            tex_form, LEFT)

        self.play(ReplacementTransform(normal_form, normal_form_dis))
        self.wait(2)

        normal_form_exp = MathTex("P(y \mid \mu)=",
                                  "\\frac{1}{\sqrt{2 \pi}}",
                                  "\exp (-\\frac{y^2}{2})",
                                  "\exp", "(",
                                  "\mu",
                                  "y",
                                  "-",
                                  "\\frac{\mu^2}{2}",
                                  ")").scale(0.8).next_to(tex_form, DOWN).align_to(tex_form, LEFT)

        self.play(ReplacementTransform(normal_form_dis, normal_form_exp))
        self.play(normal_form_exp[5].animate.set_color(GREEN))
        self.play(normal_form_exp[6].animate.set_color(BLUE))
        self.play(normal_form_exp[8].animate.set_color(YELLOW))
        self.wait(2)

        # 证明高斯分布的均值就是eta
        normal_link = MathTex('\eta = \mu').scale(0.9).next_to(self.normal_link_function, DOWN).align_to(
            self.normal_link_function, LEFT)

        self.play(FadeTransform(VGroup(normal_form_exp[5].copy(), tex_form[5].copy()), normal_link))

        self.wait(2)

        # 引入伯努利分布
        bernoulli_chart = BarChart(
            values=[0.75, 0.25],
            bar_names=["0", "1"],
            y_range=[0, 1, 10],
            y_length=4,
            x_length=5,
            x_axis_config={"font_size": 36},
            bar_colors=[WHITE, WHITE],
        ).scale(0.6).move_to(self.normal_graph).shift(DOWN)

        bernoulli_label = MathTex('\mathcal{B}(x)').next_to(bernoulli_chart, 0.001 * UP)

        self.play(FadeOut(self.normal_graph),
                  FadeOut(self.normal_link_function),
                  FadeOut(normal_link))

        self.play(FadeIn(VGroup(bernoulli_label, bernoulli_chart)))
        self.wait(1)

        bernoulli_form = MathTex('P(y \mid p)=p^y(1-p)^y').scale(0.8).next_to(tex_form, DOWN).align_to(tex_form, LEFT)

        # self.play(FadeOut(VGroup(normal_form_exp, normal_form, normal_form_dis)))
        self.play(FadeOut(normal_form_exp))

        self.play(Write(bernoulli_form))
        self.wait(2)

        bernoulli_exp = MathTex("P(y \mid p)=\exp(",
                                "\ln\\frac{p}{(1-p)}",
                                "y", "+", "\ln(1-p)\}").scale(0.8).next_to(tex_form, DOWN).align_to(tex_form, LEFT)
        self.play(Transform(bernoulli_form, bernoulli_exp))
        self.play(bernoulli_exp[1].animate.set_color(GREEN))
        self.play(bernoulli_exp[2].animate.set_color(BLUE))
        self.play(bernoulli_exp[4].animate.set_color(YELLOW))
        self.wait(3)

        bernoulli_link = MathTex('\eta =g(p)=\ln\\frac{p}{(1-p)}', color=MAROON).scale(0.9).next_to(
            self.generalized_formula[3], LEFT)

        self.play(FadeTransform(VGroup(bernoulli_exp[1].copy(), tex_form[5].copy()), bernoulli_link))

        self.wait(2)

        sigmoid_link = MathTex('p =g^{-1}(\eta) = \\frac{e^{\eta}}{1+e^\eta}', color=MAROON).scale(0.9).next_to(
            self.generalized_formula[3], LEFT)

        self.play(ReplacementTransform(bernoulli_link, sigmoid_link))
        self.wait(2)

    def normal_dis(self, x, sigma, mu):
        coef = 1 / (sigma * np.sqrt(2 * np.pi))
        expon = -1 / 2 * ((x - mu) / sigma) ** 2
        return coef * np.power(np.e, expon)


class Classification_3D(ThreeDScene):
    def construct(self):
        self.axes = None
        # self.sampler = np.random.normal(loc=1, scale=0.5, size=10)

        self.sigmoid_3D()
        # self.softmax_3D()

    def func_sigmoid(self, x_1, x_2):
        s = np.power(np.e, -(x_1 - x_2))
        return 1 / (1 + s)

    def func_softmax(self, x_1, x_2, index):
        w_1 = np.sqrt(3 / 2)
        w_2 = np.sqrt(3 / 2) * (np.sqrt(3) * x_1 - x_2) / 2
        w_3 = np.sqrt(3 / 2) * (-np.sqrt(3) * x_1 - x_2) / 2

        a, b, c = [np.power(np.e, w_1),
                   np.power(np.e, w_2),
                   np.power(np.e, w_3)]

        sum_max = np.sum(a + b + c)
        result = [a / sum_max, b / sum_max, c / sum_max]
        return result[index]

    def softmax_3D(self):
        axes = ThreeDAxes(x_range=(-6, 6), y_range=(-6, 6), z_range=(-0.15, 1.15), x_length=7, y_length=7, z_length=4)
        z_label = axes.get_z_axis_label(MathTex('Softmax'))
        self.play(Create(axes), Create(z_label))

        vg_surface = VGroup()
        for i, color in zip(range(3), [RED, BLUE, GREEN]):
            surface = Surface(lambda u, v: axes.c2p(u, v, self.func_softmax(u, v, i)),
                              u_range=[-6, 6],
                              v_range=[-6, 6],
                              resolution=(30, 30),
                              should_make_jagged=True,
                              stroke_width=0.2,
                              )
            surface.set_style(fill_opacity=0.7, stroke_color=RED)
            surface.set_fill_by_value(axes=axes, colors=[(color, 0.25), (YELLOW, 0.75), (color, 1)], axis=2)
            vg_surface.add(surface)

        self.play(Create(vg_surface))

        self.wait(3)

        self.move_camera(phi=75 * DEGREES)
        #
        # # self.set_camera_orientation(phi=75 * DEGREES, theta=0)
        # #
        # for i in range(0, 360, 90):
        #     self.move_camera(theta=i * DEGREES)

    def sigmoid_3D(self):
        axes = ThreeDAxes(x_range=(-6, 6), y_range=(-6, 6), z_range=(-0.15, 1.15), x_length=6, y_length=6, z_length=4)
        z_label = axes.get_z_axis_label(MathTex('S(x_1,x_2)'))
        self.play(Create(axes), Create(z_label))

        self.axes = VGroup(axes, z_label)

        surface_1 = Surface(lambda u, v: axes.c2p(u, v, self.func_sigmoid(u, v)),
                            u_range=[-5, 5],
                            v_range=[-5, 5],
                            resolution=(30, 30),
                            should_make_jagged=True,
                            stroke_width=0.2,
                            )

        # surface_1.set_style(fill_opacity=0.5, stroke_color=RED)

        # surface_1.set_fill_by_value(axes=axes, colors=[(RED, 0.25), (YELLOW, 0.75), (RED, 1)], axis=2)

        # surface_1.set_fill_by_checkerboard([BLUE, YELLOW, RED])

        # surface_2 = Surface(lambda u, v: axes.c2p(u, v, self.func_sigmoid(-u, -v)),
        #                     u_range=[-5, 5],
        #                     v_range=[-5, 5],
        #                     resolution=(30, 30),
        #                     should_make_jagged=True,
        #                     stroke_width=0.2,
        #                     )
        #
        # surface_2.set_style(fill_opacity=0.5, stroke_color=RED)
        #
        # surface_2.set_fill_by_value(axes=axes, colors=[(BLUE, 0.25), (YELLOW, 0.75), (BLUE, 1)], axis=2)

        self.play(Create(surface_1))  # , Create(surface_2))

        self.wait(3)

        self.move_camera(phi=75 * DEGREES)

        # self.set_camera_orientation(phi=75 * DEGREES, theta=0)
        # #
        # for i in range(0, 360, 30):
        #     self.move_camera(theta=i * DEGREES)


class Other_Regression(MovingCameraScene):
    def construct(self):
        title = Text("下一期预告").to_edge(UP)
        pass
