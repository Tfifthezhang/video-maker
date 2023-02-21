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
        #self.render_diagram()

    def write_gaussian(self):
        math_formula = MathTex('f(x)=\\frac{1}{\sigma \sqrt{2 \pi}} e^{-\\frac{1}{2}\left(\\frac{x-\mu}{\sigma}\\right)^2}').to_edge(UP+LEFT)
        self.play(Write(math_formula))

    def update_var(self):
        mu = CommonFunc.variable_tracker(label=Tex('$\mu$'), color=BLUE).scale(0.8).to_edge(LEFT)
        sigma = CommonFunc.variable_tracker(label=Tex('$\sigma$'), color=RED).scale(0.9).next_to(mu, DOWN)
        ax = CommonFunc.add_axes(x_range=[-6, 6], y_range=[0, 1], x_length=8, y_length=6)
        self.play(Create(ax))

        graph = ax.plot(lambda x: self.normal_dis(x, mu=self.mu, sigma=self.sigma), x_range=[-6, 6], use_smoothing=True)
        self.play(Create(graph))

        self.play(Create(mu), Create(sigma))

        for i, j in [[0, 3],[0, 4]]:
            self.mu = i
            self.sigma = j
            self.play(mu.tracker.animate.set_value(self.mu))
            self.play(sigma.tracker.animate.set_value(self.sigma))
            graph_other = ax.plot(lambda x: self.normal_dis(x, mu=self.mu, sigma=self.sigma), x_range=[-6, 6], use_smoothing=True)
            self.play(Transform(graph, graph_other))

    def normal_dis(self, x, sigma, mu):
        coef = 1 / (sigma * np.sqrt(2 * np.pi))
        expon = -1 / 2 * ((x - mu) / sigma) ** 2
        return coef*np.power(np.e, expon)


