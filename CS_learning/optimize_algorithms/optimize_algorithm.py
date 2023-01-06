# -*- coding: utf-8 -*-

# Copyright (C) 2021 GRGBanking All Rights Reserved

# @Time    : 2023/1/3 5:09 下午
# @Author  : 张暐（zhangwei）
# @File    : optimize_algorithm.py
# @Email   : zhangwei58@grgbanking.com
# @Software: PyCharm

from manim import *
import numpy as np
import sys

sys.path.append('..')

from CS_learning.common_func import CommonFunc


class ThreeDSurfacePlot(ThreeDScene):
    def construct(self):
        resolution_fa = 40
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        def param_gauss(u, v):
            x = u
            y = v
            sigma, mu = 0.4, [0.0, 0.0]
            d = np.linalg.norm(np.array([x - mu[0], y - mu[1]]))
            z = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
            return np.array([x, y, z])

        gauss_plane = Surface(
            param_gauss,
            resolution=(resolution_fa, resolution_fa),
            v_range=[-1.5, +1.5],
            u_range=[-1.5, +1.5]
        )

        axes = ThreeDAxes()
        gauss_plane.scale(2, about_point=ORIGIN)
        gauss_plane.set_style(fill_opacity=1, stroke_color=GREEN)
        gauss_plane.set_fill_by_value(axes=axes, colors=[(RED, 0.0), (YELLOW, 0.2), (GREEN, 0.8)], axis=2)

        grid = NumberPlane(x_range=[-10, 10, 2], y_range=[-5, 5, 2], x_length=axes.x_length, y_length=axes.y_length)

        # self.add(grid, axes, gauss_plane)
        self.play(Create(gauss_plane))


class contour_bowl(ThreeDScene):
    # CONFIG = {'inital_start': np.array([-2, 2]),
    #           'axes_x_range': [-4, 4],
    #           'axes_y_range': [-4, 4],
    #           'axes_z_range': [0, 10]}

    def construct(self):
        axes, surface = self.get_surfaceplot()
        self.move_camera(phi=45 * DEGREES, theta=0)
        l_x, l_y, l_z = self.GD(pow(2, -8) * 20, start=[-2, 2])
        arrow_vg = self.get_dot(axes, l_x, l_y, l_z)
        self.analysis(arrow_vg,surface)

    def write_text_before_animate(self, surface):
        text = MathTex('f(x,y) = x^2 +y^2').next_to(surface, UP+LEFT)
        self.play(Write(text))
        self.wait(3)
        self.play(Unwrite(text))

    def get_surfaceplot(self):

        def bowl(u, v):
            z = 1 / 2 * (np.power(u, 2) + np.power(v, 2))
            return z

        axes = ThreeDAxes(x_range=(-4, 4), y_range=(-4, 4), z_range=(0, 8)).shift(np.array([0, 0, -2]))
        surface_plane = Surface(lambda u, v: axes.c2p(u, v, bowl(u, v)),
                                u_range=[-2, 2],
                                v_range=[-2, 2],
                                resolution=(30, 30),
                                should_make_jagged=True,
                                stroke_width=0.2)

        surface_plane.set_style(fill_opacity=0.5, stroke_color=RED)

        surface_plane.set_fill_by_value(axes=axes, colors=[(RED, 0.0), (YELLOW, 0.2), (BLUE, 4)], axis=2)

        self.add(axes)
        self.play(Create(surface_plane))

        self.write_text_before_animate(surface_plane)

        self.move_camera(phi=75 * DEGREES)

        #self.set_camera_orientation(phi=75 * DEGREES, theta=0)

        for i in range(0, 120, 30):
            self.move_camera(theta=i * DEGREES)

        return axes, surface_plane

    @staticmethod
    def f(x, y):
        z = 1 / 2 * (np.power(x, 2) + np.power(y, 2))
        return np.array([x, y, z])

    def write_function(self, surface):
        tex = MathTex('f(x,y) = x^2+y^2').scale(0.6).next_to(surface, UP + LEFT)
        self.play(Create(tex))

    def GD(self, lr, start):
        def partial_x(x):
            return x

        def partial_y(y):
            return y

        x, y = start[0], start[1]
        GD_x, GD_y, GD_z = [], [], []
        for it in range(50):
            GD_x.append(x)
            GD_y.append(y)
            GD_z.append(self.f(x, y)[-1])
            dx = partial_x(x)
            dy = partial_y(y)
            x = x - lr * dx
            y = y - lr * dy
        return GD_x, GD_y, GD_z

    def get_dot(self, axes, l_x, l_y, l_z):
        point = Dot3D(axes.c2p(l_x[0], l_y[0], l_z[0]), radius=0.1, color=BLACK)
        self.play(Create(point))
        arrow_vg = VGroup()
        for i in range(1, len(l_x)):
            arrow = Arrow(start=axes.c2p(l_x[i - 1], l_y[i - 1], l_z[i - 1]),
                          end=axes.c2p(l_x[i], l_y[i], l_z[i]), color=BLACK, stroke_width=5)
            self.add(arrow)
            self.play(point.animate.move_to(axes.c2p(l_x[i], l_y[i], l_z[i])))

            arrow_vg.add(arrow)
        return arrow_vg

    def analysis(self, arrow_vg, surface):
        #self.move_camera(phi=0, theta=0)
        self.set_camera_orientation(phi=0, theta=0)
        s = len(arrow_vg)
        var = CommonFunc.variable_tracker(label=Text('迭代次数'), start=0, color=RED).scale(0.6).next_to(surface, DOWN+RIGHT)
        self.add(var)
        self.play(var.tracker.animate.set_value(s))


class saddle_point(ThreeDScene):
    def construct(self):
        axes, surface = self.get_surfaceplot()

    def get_surfaceplot(self):
        def bowl(u, v):
            z = np.sin(2 * u) * np.cos(2 * v) + 3
            return z

        axes = ThreeDAxes(x_range=(-4, 4), y_range=(-4, 4), z_range=(0, 8)).shift(np.array([0, 0, -2]))
        surface_plane = Surface(lambda u, v: axes.c2p(u, v, bowl(u, v)),
                                u_range=[-3, 3],
                                v_range=[-3, 3],
                                resolution=(30, 30),
                                should_make_jagged=True,
                                stroke_width=0.2)

        surface_plane.set_style(fill_opacity=1, stroke_color=RED)

        surface_plane.set_fill_by_value(axes=axes, colors=[(RED, 0.0), (YELLOW, 0.2), (BLUE, 4)], axis=2)

        self.set_camera_orientation(phi=75 * DEGREES, theta=0)

        self.add(axes)
        self.play(Create(surface_plane))

        # for i in range(0, 360, 15):
        #     self.move_camera(theta=i * DEGREES)

        self.move_camera(phi=45 * DEGREES, theta=0)

        return axes, surface_plane
