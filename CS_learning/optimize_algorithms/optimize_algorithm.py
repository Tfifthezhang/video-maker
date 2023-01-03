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


class logo(Scene):
    def construct(self):
        phanton = CommonFunc.add_function(lambda x: 0.3 * np.sin(5 * x), x_range=(-3, 3))
        start, end = phanton.get_start(), phanton.get_end()

        e_minus = CommonFunc.add_arrow(np.array([-5, 2, 0]), start,
                                       color=RED, max_tip_length_to_length_ratio=0.05)
        e_plus = CommonFunc.add_arrow(start, np.array([-5, -2, 0]),
                                      color=RED, max_tip_length_to_length_ratio=0.05)

        q_average = CommonFunc.add_arrow(np.array([5, 2, 0]), end,
                                         color=GREEN, max_tip_length_to_length_ratio=0.05)
        q = CommonFunc.add_arrow(end, np.array([5, -2, 0]),
                                 color=GREEN, max_tip_length_to_length_ratio=0.05)

        self.play(GrowArrow(e_minus))
        self.play(GrowArrow(e_plus))
        self.wait()
        self.play(Create(phanton))
        self.wait()
        self.play(GrowFromPoint(q_average, end))
        self.play(GrowFromPoint(q, end))

        group = VGroup(phanton, e_minus, e_plus, q_average, q)

        self.play(group.animate.scale(0.6))

        phanton_group = VGroup(*[CommonFunc.add_function(lambda x: 0.3 * np.sin(5 * x - np.pi * 0.5 * dx),
                                                         x_range=(-3, 3)).scale(0.6) for dx in range(41)])

        text_cn = Text('迷路的小画家', font='HiraginoSansGB-W3').scale(0.7).next_to(group, DOWN)

        self.play(FadeIn(text_cn), run_time=3)

        svg_image = SVGMobject('svg_icon/bird.svg', fill_color=MAROON).scale(1).next_to(phanton, 2 * UP)
        self.play(SpinInFromNothing(svg_image))

        self.play(Succession(*[Transform(phanton, phanton_group[i]) for i in range(len(phanton_group))], run_time=5))


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


class contour_bowl(ThreeDSurfacePlot):
    # CONFIG = {'inital_start': np.array([-2, 2]),
    #           'axes_x_range': [-4, 4],
    #           'axes_y_range': [-4, 4],
    #           'axes_z_range': [0, 10]}

    def construct(self):
        title = Title('$f(x,y) = x^2 +y^2$')
        self.play(Write(title))
        self.wait(3)
        self.play(Unwrite(title))
        axes, surface = self.get_surfaceplot()
        l_x, l_y, l_z = self.GD(pow(2, -6)*10, start=[-2, 2])
        arrows = self.get_dot(axes, l_x, l_y, l_z)
        self.analysis()

    def get_surfaceplot(self):

        def bowl(u, v):
            z = 1/2 * (np.power(u, 2) + np.power(v, 2))
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

        self.set_camera_orientation(phi=75 * DEGREES, theta=0)

        self.add(axes)
        self.play(Create(surface_plane))

        for i in range(0, 360, 30):
            self.move_camera(theta=i * DEGREES)

        self.move_camera(phi=45 * DEGREES, theta=0)

        return axes, surface_plane

    @staticmethod
    def f(x, y):
        z = 1 / 2 * (np.power(x, 2) + np.power(y, 2))
        return np.array([x, y, z])

    def write_function(self, surface):
        tex = MathTex('f(x,y) = x^2+y^2').scale(0.6).next_to(surface, UP+LEFT)
        self.play(Create(tex))

    def GD(self, lr, start):
        def partial_x(x):
            return 1/2 * x

        def partial_y(y):
            return 1/2 * y

        x, y = start[0], start[1]
        GD_x, GD_y, GD_z = [], [], []
        for it in range(10):
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
            arrow = Arrow(start=axes.c2p(l_x[i-1], l_y[i-1], l_z[i-1]),
                          end=axes.c2p(l_x[i], l_y[i], l_z[i]), color=GREEN, stroke_width=5)
            self.add(arrow)
            self.play(point.animate.move_to(axes.c2p(l_x[i], l_y[i], l_z[i])))

            arrow_vg.add(arrow)
        return arrow_vg

    def analysis(self):
        self.move_camera(phi=0, theta=0)


class saddle_point(ThreeDScene):
    def construct(self):
        axes, surface = self.get_surfaceplot()

    def get_surfaceplot(self):
        def bowl(u, v):
            z = np.sin(2*u)*np.cos(2*v)+3
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

