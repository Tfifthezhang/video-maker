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


class NormalDistribution(Scene):
    def construct(self):
        self.sigma = 1
        self.mu = 0

        self.write_gaussian()
        self.update_var()
        # self.render_diagram()

    def write_gaussian(self):
        math_formula = MathTex(
            'f(x)=\\frac{1}{\sigma \sqrt{2 \pi}} e^{-\\frac{1}{2}\left(\\frac{x-\mu}{\sigma}\\right)^2}').to_edge(
            UP + LEFT)
        self.play(Write(math_formula))

    def update_var(self):
        mu = CommonFunc.variable_tracker(label=Tex('$\mu$'), color=BLUE).scale(0.8).to_edge(LEFT)
        sigma = CommonFunc.variable_tracker(label=Tex('$\sigma$'), color=RED).scale(0.9).next_to(mu, DOWN)
        ax = CommonFunc.add_axes(x_range=[-6, 6], y_range=[0, 1], x_length=8, y_length=6)
        self.play(Create(ax))

        graph = ax.plot(lambda x: self.normal_dis(x, mu=self.mu, sigma=self.sigma), x_range=[-6, 6], use_smoothing=True)
        self.play(Create(graph))

        self.play(Create(mu), Create(sigma))

        for i, j in [[0, 3], [0, 4]]:
            self.mu = i
            self.sigma = j
            self.play(mu.tracker.animate.set_value(self.mu))
            self.play(sigma.tracker.animate.set_value(self.sigma))
            graph_other = ax.plot(lambda x: self.normal_dis(x, mu=self.mu, sigma=self.sigma), x_range=[-6, 6],
                                  use_smoothing=True)
            self.play(Transform(graph, graph_other))

    def normal_dis(self, x, sigma, mu):
        coef = 1 / (sigma * np.sqrt(2 * np.pi))
        expon = -1 / 2 * ((x - mu) / sigma) ** 2
        return coef * np.power(np.e, expon)


class Regression(MovingCameraScene):
    def construct(self):
        self.plot_scatter()

    def plot_scatter(self):
        ax = CommonFunc.add_axes(x_range=[-8, 8], y_range=[-8, 8], x_length=8, y_length=6,
                                 axis_config={"include_tip": False, "include_numbers": False})
        self.play(Create(ax))

        x = np.linspace(-7.5, 7.5, 200)
        gaussian_noise = np.random.normal(size=(200,))
        y = np.power(0.25 * x, 3)
        y_noise = y + gaussian_noise
        coords = list(zip(x, y_noise))

        # coords = np.random.uniform(-8, 8, size=(10, 2))

        dots = VGroup(
            *[Dot(ax.c2p(coord[0], coord[1]), radius=0.5 * DEFAULT_DOT_RADIUS, color=BLUE) for coord in coords])
        self.play(FadeIn(dots))

        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.scale(0.3).move_to(dots[140]))

        self.play(dots[140].animate.set(color=RED))

        iid_axes = Axes(x_range=[-3, 3], y_range=[0, 0.6], x_length=5, y_length=1,
                        axis_config=dict(include_tip=False,
                                         include_numbers=False,
                                         rotation=0 * DEGREES,
                                         stroke_width=1.0),).scale(0.3).rotate(270*DEGREES).next_to(dots[140], 0.05*RIGHT)
        self.play(Create(iid_axes))

        #self.play(iid_axes.animate.rotate(270*DEGREES))
        graph = iid_axes.plot(lambda x: self.normal_dis(x, mu=0, sigma=1),
                              x_range=[-3, 3],
                              use_smoothing=True,
                              color=MAROON)
        self.play(Create(graph))

        self.wait(3)
        self.play(Restore(self.camera.frame))


    def normal_dis(self, x, sigma, mu):
        coef = 1 / (sigma * np.sqrt(2 * np.pi))
        expon = -1 / 2 * ((x - mu) / sigma) ** 2
        return coef * np.power(np.e, expon)
