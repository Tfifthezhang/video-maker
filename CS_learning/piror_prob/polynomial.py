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
        pass

    @staticmethod
    def true_fun(X):
        return np.cos(1.5 * np.pi * X)

    def X_Y(self, n_samples):
        np.random.seed(42)
        X = np.sort(np.random.rand(n_samples))
        y = self.true_fun(X) + np.random.randn(n_samples) * 0.1

        X_test = np.linspace(0, 1, 200)
        y_test = self.true_fun(X_test)
        return X, y, X_test, y_test

    def poly_regression(self):
        degrees = [1, 4, 15]
        X, y, X_test, y_test = self.X_Y(30)

        ax = CommonFunc.add_axes(x_range=[0, 1], y_range=[-1, 1], x_length=8, y_length=6,
                                 axis_config={"include_tip": False, "include_numbers": False})
        self.play(Create(ax))

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
            y_predict = pipeline.predict(X_test[:, np.newaxis])












