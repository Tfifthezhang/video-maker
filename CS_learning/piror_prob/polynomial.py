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

        self.data_prepare()
        self.poly_reagression()

    @staticmethod
    def true_fun(X):
        return np.cos(1.5 * np.pi * X)

    def X_Y(self, n_samples):
        np.random.seed(42)
        X = np.sort(np.random.rand(n_samples))
        y = self.true_fun(X) + np.random.randn(n_samples) * 0.1

        X_test = np.linspace(0, 1, 200)
        y_test = self.true_fun(X_test)
        gaussian_noise = np.random.normal(loc=0, scale=0.1, size=(200,))
        y_test = gaussian_noise+y_test
        return X, y, X_test, y_test

    def data_prepare(self):
        X, y, X_test, y_test = self.X_Y(30)

        ax = CommonFunc.add_axes(x_range=[-0.1, 1], y_range=[-1, 1], x_length=8, y_length=6,
                                 axis_config={"include_tip": False, "include_numbers": False})
        self.play(Create(ax))
        self.poly_ax.add(ax)

        # axes_labels = ax.get_axis_labels(x_label=MathTex('x'), y_label=MathTex('y'))
        # self.play(Create(axes_labels))
        # self.poly_ax.add(axes_labels)

        coords = list(zip(X_test, y_test))

        dots = VGroup(
            *[Dot(ax.c2p(coord[0], coord[1]), radius=0.5 * DEFAULT_DOT_RADIUS, color=BLUE) for coord in coords])
        self.poly_ax.add(dots)
        self.play(FadeIn(dots))

        self.wait(2)

    def poly_reagression(self):
        ax = self.poly_ax[0]
        degrees = [1, 4, 15]
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
            pipeline.fit(X_test[:, np.newaxis], y_test)
            y_predict = pipeline.predict(X_test[:, np.newaxis])
            inital_plot = ax.plot(X_test, y_predict, x_range=[-0.1, 1], use_smoothing=True, color=MAROON)
            self.play(Create(inital_plot))
            self.wait(1)














