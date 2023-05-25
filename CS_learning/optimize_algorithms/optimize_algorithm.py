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


class normal_neassery(ThreeDScene):

    def construct(self):
        axes, surface = self.get_surfaceplot()
        self.move_camera(phi=45 * DEGREES, theta=0)
        # l_x, l_y, l_z = self.GD(pow(2, -8) * 20, start=[-2, 2])
        # arrow_vg = self.get_dot(axes, l_x, l_y, l_z)
        # self.analysis(arrow_vg,surface)

    def write_text_before_animate(self, surface):
        text = MathTex('f(x,y) = x^2 +y^2').next_to(surface, UP + LEFT)
        self.play(Write(text))
        self.wait(3)
        self.play(Unwrite(text))

    def get_surfaceplot(self):

        def bowl(u, v):
            z = 1 / 2 * (np.power(u, 2) + np.power(v, 2) / 5)
            return z

        axes = ThreeDAxes(x_range=(-4, 4), y_range=(-2, 2), z_range=(0, 8)).shift(np.array([0, 0, -2]))
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

        # self.set_camera_orientation(phi=75 * DEGREES, theta=0)

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
        # self.move_camera(phi=0, theta=0)
        self.set_camera_orientation(phi=0, theta=0)
        s = len(arrow_vg)
        var = CommonFunc.variable_tracker(label=Text('迭代次数'), start=0, color=RED).scale(0.6).next_to(surface,
                                                                                                     DOWN + RIGHT)
        self.add(var)
        self.play(var.tracker.animate.set_value(s))


class Taylor_Expan(Scene):
    def construct(self):
        self.tylor_tex = None
        self.func_ax = None

        self.taylor_write()
        self.graph_axes()
        self.graph_order()

    def taylor_write(self):
        tylor = MathTex("f(x)=", "\\frac{f(x_0)}{0 !}", "+",
                        "\\frac{f^{\prime}(x_0)}{1 !}","(x-x_0)","+",
                        "\\frac{f^{\prime\prime}(x_0)}{2 !}(x-x_0)^2", "+",
                        "\ldots", "+",
                        "\\frac{f^{(n)}(x_0)}{n !}(x-x_0)^n", "+",
                        "R_n(x)").scale(0.6)

        self.play(Write(tylor))
        self.wait(2)

        self.tylor_tex = tylor

        self.play(tylor.animate.to_edge(UP))

    def graph_axes(self):
        ax = CommonFunc.add_axes(x_range=[-10, 10], y_range=[-1, 1], x_length=7, y_length=2,
                                 axis_config={"include_tip": False, "include_numbers": False}).scale(0.9)

        square_plot = ax.plot(lambda x:  np.sin(x), x_range=[-10, 10], use_smoothing=True, color=WHITE)
        square_label = ax.get_graph_label(graph=square_plot, label=MathTex('\sin(x)').scale(0.8), x_val=12, direction= UP / 2)

        self.func_ax = VGroup(ax, square_plot, square_label)

        self.play(Create(self.func_ax))
        self.wait(2)

    def graph_order(self):
        ax = self.func_ax[0]

        # 零阶
        order0_plot = ax.plot(lambda x: self.sin_0(1), x_range=[-10, 10], use_smoothing=True, color=RED)

        self.play(Create(order0_plot))
        self.wait(2)

        # 一阶
        order1_plot = ax.plot(lambda x: self.sin_0(1)+self.sin_1(1)*(x-1), x_range=[-10, 10], use_smoothing=True, color=BLUE)

        self.play(FadeTransform(order0_plot, order1_plot))
        self.wait(2)

        # 二阶
        order2_plot = ax.plot(lambda x: self.sin_0(1) + self.sin_1(1)*(x-1)
                                        + 0.5*self.sin_2(1)*np.power(x-1, 2),
                              x_range=[-10, 10], use_smoothing=True, color=GREEN)

        self.play(FadeTransform(order1_plot, order2_plot))
        self.wait(2)

        # 三阶

        order3_plot = ax.plot(lambda x: self.sin_0(1) + self.sin_1(1)*(x-1)
                                        + 0.5*self.sin_2(1)*np.power(x-1, 2)
                                        + 1/6*self.sin_3(1)*np.power(x-1, 3),
                              x_range=[-10, 10], use_smoothing=True, color=MAROON)

        self.play(FadeTransform(order2_plot, order3_plot))
        self.wait(2)

        # 四阶
        order4_plot = ax.plot(lambda x: self.sin_0(1) + self.sin_1(1) * (x - 1)
                                        + 0.5 * self.sin_2(1) * np.power(x - 1, 2)
                                        + 1 / 6 * self.sin_3(1) * np.power(x - 1, 3)
                                        + 1 / 24 * self.sin_0(1) * np.power(x - 1, 4),
                              x_range=[-10, 10], use_smoothing=True, color=PURE_RED)

        self.play(FadeTransform(order3_plot, order4_plot))
        self.wait(2)

    def graph_tex(self):
        # 开始迭代box，说明概率和对数概率协调更新
        vg_box = VGroup(*[SurroundingRectangle(self.entropy_example[i], buff=.05) for i in [1, 3, 5, 7]])
        t_labels = VGroup(
            *[log_ax.get_T_label(x_val=i, graph=self.vg_graph[1], label=MathTex(str(i))).scale(0.9) for i in
              [1 / 2, 1 / 4, 1 / 8, 1 / 8]])
        box = vg_box[0]
        t_label = t_labels[0]
        self.play(FadeIn(box), Write(t_label))

        for i in range(1, len(vg_box)):
            self.play(Transform(box, vg_box[i]),
                      Transform(t_label, t_labels[i]))
            self.wait(1)

        self.play(FadeOut(box), FadeOut(t_label))

        self.wait(2)

    @staticmethod
    def sin_0(x0):
        return np.sin(x0)

    @staticmethod
    def sin_1(x):
        return np.cos(x)

    @staticmethod
    def sin_2(x):
        return -np.sin(x)

    @staticmethod
    def sin_3(x):
        return -np.cos(x)





class Opt_gradient(Scene):
    def construct(self):
        self.text = None
        self.ax = VGroup()
        self.m = 10
        self.loss_track = None
        self.loss_graph = None
        self.iterations = None
        self.learning_rate = [0.5, 2, 2.05]
        self.lr_text = None

        self.get_surfaceplot()
        self.get_dot()

    def coordinate_desent(self, lr, start):
        def partial_x(x):
            return x

        def partial_y(y):
            return y

        x, y = start[0], start[1]
        GD_x, GD_y, GD_z = [], [], []
        for it in range(20):
            GD_x.append(x)
            GD_y.append(y)
            GD_z.append(self.f(x, y)[-1])
            dx = partial_x(x)
            dy = partial_y(y)
            if it % 2 == 0:
                x = x - lr * dx
            else:
                y = y - lr * dy
        return GD_x, GD_y, GD_z

    @staticmethod
    def f(x, y):
        z = 1 / 2 * (np.power(x, 2) + np.power(y, 2))
        return np.array([x, y, z])

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

        for i in range(6):
            self.play(Rotate(self.ax, angle=60 * DEGREES, axis=UP))

        self.play(Rotate(self.ax, angle=15 * DEGREES, axis=UP))
        self.play(Rotate(self.ax, angle=45 * DEGREES, axis=RIGHT))

    def get_dot(self):
        l_x, l_y, l_z = self.coordinate_desent(lr=0.1, start=[2, 2])
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


class Opt_VF(Scene):
    def construct(self):
        func = lambda pos: pos+LEFT*2
        colors = [RED, YELLOW, BLUE, DARK_GRAY]
        # min_radius = Circle(radius=2, color=colors[0]).shift(LEFT * 5)
        # max_radius = Circle(radius=10, color=colors[-1]).shift(LEFT * 5)
        vf = ArrowVectorField(func, min_color_scheme_value=2, max_color_scheme_value=10, colors=colors).scale(0.4)
        self.play(Write(vf))
        self.wait(2)
        #self.add(vf, min_radius, max_radius)


class Opt_GD(ThreeDScene):
    def construct(self):
        self.text = None
        self.ax = VGroup()
        self.m = 10
        self.loss_track = None
        self.loss_graph = None
        self.iterations = None
        self.learning_rate = [0.5, 2, 2.05]
        self.lr_text = None

        self.write_text_before_animate()
        self.get_surfaceplot()
        self.write_lr()
        self.get_dot()

        # self.move_camera(phi=45 * DEGREES, theta=0)
        # l_x, l_y, l_z = self.GD(pow(2, -8) * 20, start=[-2, 2])

    def write_text_before_animate(self):
        text = MathTex('\mathcal{L}(\omega_1, \omega_2) = \omega_1^2 +\omega_2^2').to_edge(LEFT + UP)
        text_gradient = MathTex("\mathbf{\omega}_{m+1} =",
                                "\mathbf{\omega}_m - ",
                                "\epsilon",
                                "\\nabla \mathcal{L}(\mathbf{\omega}_m)").to_edge(LEFT + UP)
        text_gradient[2].set_color(MAROON)
        self.play(Write(text_gradient))
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

    def write_lr(self):
        tex_1 = MathTex('\epsilon < 2')
        tex_2 = MathTex('\epsilon = 2')
        tex_3 = MathTex('\epsilon > 2')

        vg_tex = VGroup(tex_1, tex_2, tex_3)

        vg_tex.arrange_submobjects(DOWN, buff=3.5).scale(0.65).to_edge(3.5 * RIGHT)

        # self.play(Write(vg_tex))
        # self.wait(2)

        self.lr_text = vg_tex

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
            self.play(Write(self.lr_text[lr]))
            self.wait(2)

            l_x, l_y, l_z = self.GD(self.learning_rate[lr], start=[2, 2])

            axes = self.ax[0]

            point = Dot3D(axes.c2p(l_x[0], l_y[0], l_z[0]), radius=0.1, color=MAROON)
            self.play(FadeTransform(self.lr_text[lr].copy(), point))
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
                                     axis_config={"include_tip": False, "include_numbers": False}).scale(0.5).move_to(
                self.lr_text[lr].get_center())

            labels = ax.get_axis_labels(Text("m").scale(0.5), MathTex("\mathcal{L}").scale(0.45))

            gd_plot = ax.plot_line_graph(x_values=list(range(self.m)), y_values=self.loss_track[-1],
                                         line_color=GOLD_E,
                                         vertex_dot_style=dict(stroke_width=1, fill_color=PURPLE),
                                         stroke_width=3)
            # gd_label = ax.get_graph_label(graph=gd_plot, label=MathTex('\\text{lr}=2').scale(0.7), direction=UP)

            self.loss_graph = VGroup(ax, labels, gd_plot)

            self.play(FadeTransform(self.lr_text[lr], self.loss_graph))

            self.wait(2)

            self.play(FadeOut(self.iterations))


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
