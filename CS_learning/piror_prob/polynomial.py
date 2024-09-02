# -*- coding: utf-8 -*-
from manim import *
import numpy as np
import sys

sys.path.append('..')

from CS_learning.common_func import CommonFunc
from sklearn.datasets import make_moons, make_blobs, make_classification
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

class PolynomialRegression(Scene):
    def construct(self):
        self.poly_ax =VGroup()
        self.linear_formula = None
        self.degree= None

        self.data_prepare()
        self.linear_example()
        self.mse_dot()
        self.poly_formula()
        self.poly_reagression()

    @staticmethod
    def true_fun(X):
        return np.cos(1.5 * np.pi * X)

    @staticmethod
    def alay_formula(l_coef, inter):
        length = len(l_coef)
        s_init = str(inter)
        for i in range(length):
            s = '{}*x**{}'.format(l_coef[i], i+1)
            s_init += '+' + s
        return s_init

    def get_vertical_line(self, ax, dots, X, graph):
        vg_diff_line = VGroup()
        for i in range(len(dots)):
            dot = dots[i]
            graph_dot = ax.input_to_graph_point(X[i], graph=graph)
            vg_diff_line.add(CommonFunc.add_line(dot, graph_dot, color=RED))
        return vg_diff_line

    def X_Y(self, n_samples):
        np.random.seed(0)
        X = np.sort(np.random.rand(n_samples))
        y = self.true_fun(X) + np.random.randn(n_samples) * 0.1

        X_test = np.linspace(0, 1, 200)
        y_test = self.true_fun(X_test)
        gaussian_noise = np.random.normal(loc=0, scale=0.1, size=(200,))
        y_test = gaussian_noise+y_test
        return X, y, X_test, y_test

    def data_prepare(self):
        X, y, X_test, y_test = self.X_Y(30)

        ax = CommonFunc.add_axes(x_range=[-0.1, 1], y_range=[-1.5, 1.5], x_length=8, y_length=6,
                                 axis_config={"include_tip": False, "include_numbers": False}).scale(0.9)
        self.play(Create(ax))
        self.poly_ax.add(ax)

        coords = list(zip(X, y))

        dots = VGroup(
            *[Dot(ax.c2p(coord[0], coord[1]), radius=0.8 * DEFAULT_DOT_RADIUS, color=BLUE) for coord in coords])
        self.poly_ax.add(dots)
        self.play(FadeIn(dots))

        self.wait(2)

    def linear_example(self):
        ax, dots = self.poly_ax[0], self.poly_ax[1]
        X, y, X_test, y_test = self.X_Y(30)

        s_function = '0.5366803303178728 + -1.6093117914612283 * x ** 1'
        linear_rg = MathTex('y = \omega x + b').to_edge(UP + LEFT)
        self.play(Write(linear_rg))
        self.linear_formula = linear_rg

        init_linear = ax.plot(lambda x: eval(s_function), x_range=[-0.1, 1, 0.005], use_smoothing=True, color=YELLOW)
        self.play(Create(init_linear))
        self.wait(1)

        vg_diff_line = self.get_vertical_line(ax, dots, X, init_linear)

        self.play(Create(vg_diff_line))

        self.wait(1)

        self.inital_plot = init_linear
        self.vg_diff_line = vg_diff_line

    def mse_dot(self):
        mse_ax = CommonFunc.add_axes(x_range=[0, 20], y_range=[0, 25], x_length=6, y_length=4,
                                 axis_config={"include_tip": False, "include_numbers": False}).scale(0.45).to_edge(RIGHT)
        mse_formula = Text('均方误差MSE').scale(0.45).next_to(mse_ax, UP)
        path = VMobject()
        dot = Dot(mse_ax.c2p(0, 25), radius=DEFAULT_DOT_RADIUS, color=RED)
        path.set_points_as_corners([dot.get_center(), dot.get_center()])

        def update_path(path):
            previous_path = path.copy()
            previous_path.add_points_as_corners([dot.get_center()])
            path.become(previous_path)

        path.add_updater(update_path)

        self.play(FadeIn(mse_ax),Create(mse_formula),
                  FadeTransform(self.vg_diff_line, dot))
        self.add(path)

        self.vg_mse = VGroup(mse_ax, dot, path, mse_formula)

    def poly_formula(self):
        poly_rg = MathTex("y = \omega_n x^", "n", "+ \omega_{n-1}x^{n-1}+.....+ \omega_1x+b").to_edge(UP + LEFT)
        self.play(FadeTransformPieces(self.linear_formula, poly_rg))
        self.wait(1)

        self.linear_formula = poly_rg

        self.play(FocusOn(poly_rg[1]))
        self.wait(1)

        poly_degree = Variable(1, Text("最高阶n").scale(0.65), var_type=Integer).to_edge(LEFT)
        poly_degree.label.set_color(MAROON)

        self.play(FadeIn(poly_degree, target_position=poly_rg[1]))
        self.wait(2)

        self.degree = poly_degree

    def poly_reagression(self):
        ax, dots = self.poly_ax[0], self.poly_ax[1]
        degrees = range(2, 20)
        X, y, X_test, y_test = self.X_Y(30)

        for i in range(len(degrees)):
            polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
            linear_regression = LinearRegression()
            pipeline = Pipeline(
                [
                    ("polynomial_features", polynomial_features),
                    ("linear_regression", linear_regression),
                ]
            )
            pipeline.fit(X[:, np.newaxis], y)
            l_coef, inter = pipeline[-1].coef_, pipeline[-1].intercept_
            formula = self.alay_formula(l_coef, inter)
            if degrees[i]==18:
                continue

            fit_plot = ax.plot(lambda x: eval(formula), x_range=[-0.1, 1, 0.005], use_smoothing=True, color=YELLOW)
            vg_diff_fit_line = self.get_vertical_line(ax, dots, X, fit_plot)

            self.play(self.degree.tracker.animate.set_value(degrees[i]))
            self.play(Transform(self.inital_plot, fit_plot),
                      Transform(self.vg_diff_line, vg_diff_fit_line))
            self.play(self.vg_mse[1].animate.move_to(self.vg_mse[0].c2p(i+1, (5-0.26*(i+1))**2)))
            self.wait(1)

