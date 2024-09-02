# -*- coding: utf-8 -*-
from manim import *
import numpy as np
import sys

sys.path.append('..')

from CS_learning.common_func import CommonFunc

class BirthdayParadox(Scene):
    def construct(self):
        self.poly_ax =VGroup()
        self.linear_formula = None
        self.degree= None

        self.intro_problem()
        #self.data_prepare()
        #self.mse_dot()

    @staticmethod
    def birth_func(k, n):
        up = k*(k-1)
        down = 2*n
        return 1-np.exp(-up/down)

    def intro_problem(self):
        n_svg = 24
        vg_svg = VGroup(*[SVGMobject('svg_icon/people.svg', fill_color=WHITE).scale(0.6) for _ in range(n_svg)])

        vg_svg.arrange_in_grid(rows=4, buff=0.35)

        self.play(Create(vg_svg))
        self.wait(2)

    def data_prepare(self):
        ax = CommonFunc.add_axes(x_range=[0, 101], y_range=[0, 1.0], x_length=8, y_length=6,
                                 axis_config={"include_tip": False, "include_numbers": False}).scale(1)
        self.play(Create(ax))
        self.poly_ax.add(ax)
        fit_plot = ax.plot(lambda x: self.birth_func(k=x, n=365), x_range=[1, 100], use_smoothing=True, color=YELLOW)

        self.play(Create(fit_plot))
        self.wait(2)

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
        self.play(self.vg_mse[1].animate.move_to(self.vg_mse[0].c2p(i + 1, (5 - 0.26 * (i + 1)) ** 2)))


