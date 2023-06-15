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
import os

sys.path.append('..')

from CS_learning.common_func import CommonFunc


class ArgMin(Scene):
    def construct(self):
        self.write_graph()

    def write_graph(self):

        ax = Axes(x_range=[0, 10], y_range=[0, 25], axis_config={"include_tip": False}).scale(0.8)
        labels = ax.get_axis_labels(x_label="\omega", y_label="\mathcal{L}")

        t = ValueTracker(0)

        def func(x):
            return (x - 5) ** 2

        graph = ax.plot(func, color=MAROON)

        initial_point = [ax.coords_to_point(t.get_value(), func(t.get_value()))]
        dot = Dot(point=initial_point)

        dot.add_updater(lambda x: x.move_to(ax.c2p(t.get_value(), func(t.get_value()))))
        x_space = np.linspace(*ax.x_range[:2], 200)
        minimum_index = func(x_space).argmin()

        vg_graph = VGroup(ax, labels, graph, dot)

        self.play(Create(vg_graph))
        self.wait(2)


        final = Vector(DOWN).next_to(ax.coords_to_point(5, 0), UP)
        self.play(FadeIn(final))
        self.wait(2)


        gradient_descent = MathTex("\omega_{t+1}", "=", "\omega_{t}-",
                                   "\epsilon","\mathbf{g}(\omega_{t})").next_to(graph, UP)

        gradient = MathTex('\mathbf{g}(\omega_{t}) = \\nabla \mathcal{L}(\omega_{t})').scale(0.7).next_to(gradient_descent, DOWN)

        self.play(Create(gradient_descent), Create(gradient))

        self.play(t.animate.set_value(x_space[minimum_index]))
        self.wait(2)

        self.play(Circumscribe(VGroup(gradient_descent)))
        self.wait(2)


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
        self.tylor_example = None
        self.anchor_label = None

        self.taylor_write()
        self.graph_axes()
        self.graph_order()
        self.graph_0()
        self.graph_0_plot()

    def taylor_write(self):
        tylor = MathTex("f(x)=", "\\frac{f(x_0)}{0 !}", "+",
                        "\\frac{f^{\prime}(x_0)}{1 !}", "(x-x_0)", "+",
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
                                 axis_config={"include_tip": False, "include_numbers": False}).scale(0.95).shift(
            0.5 * DOWN)

        square_plot = ax.plot(lambda x: np.sin(x), x_range=[-10, 10], use_smoothing=True, color=WHITE)
        square_label = ax.get_graph_label(graph=square_plot, label=MathTex('\sin(x)').scale(0.8), x_val=12,
                                          direction=UP / 2)

        self.func_ax = VGroup(ax, square_plot, square_label)

        self.play(Create(self.func_ax))
        self.wait(2)

        x1_label = ax.get_T_label(x_val=1, graph=square_plot, label=MathTex(1)).scale(0.9)
        self.play(Write(x1_label))
        self.wait(1)
        self.anchor_label = x1_label

        sin_tylor = MathTex("\sin(x)=", "\\frac{f(1)}{0 !}", "+",
                            "\\frac{f^{\prime}(1)}{1 !}(x-1)", "+",
                            "\\frac{f^{\prime\prime}(1)}{2 !}(x-1)^2", "+",
                            "\\frac{f^{\prime\prime\prime}(1)}{3 !}(x-1)^3", "+",
                            "\\frac{f^{\prime\prime\prime\prime}(1)}{4 !}(x-1)^4", "+",
                            "\ldots", ).scale(0.5).next_to(self.tylor_tex, 2 * DOWN)

        self.play(FadeTransform(self.tylor_tex.copy(), sin_tylor))
        self.wait(2)

        self.tylor_example = sin_tylor

    def graph_order(self):
        vg_box = VGroup(*[SurroundingRectangle(self.tylor_example[i], buff=.05) for i in [1, 3, 5, 7, 9]])

        ax = self.func_ax[0]

        # 零阶
        order0_plot = ax.plot(lambda x: self.sin_0(1), x_range=[-10, 10], use_smoothing=True, color=RED)

        # 一阶
        order1_plot = ax.plot(lambda x: self.sin_0(1) + self.sin_1(1) * (x - 1), x_range=[-10, 10], use_smoothing=True,
                              color=BLUE)

        # 二阶
        order2_plot = ax.plot(lambda x: self.sin_0(1) + self.sin_1(1) * (x - 1)
                                        + 0.5 * self.sin_2(1) * np.power(x - 1, 2),
                              x_range=[-10, 10], use_smoothing=True, color=GREEN)

        # 三阶

        order3_plot = ax.plot(lambda x: self.sin_0(1) + self.sin_1(1) * (x - 1)
                                        + 0.5 * self.sin_2(1) * np.power(x - 1, 2)
                                        + 1 / 6 * self.sin_3(1) * np.power(x - 1, 3),
                              x_range=[-10, 10], use_smoothing=True, color=MAROON)

        # 四阶
        order4_plot = ax.plot(lambda x: self.sin_0(1) + self.sin_1(1) * (x - 1)
                                        + 0.5 * self.sin_2(1) * np.power(x - 1, 2)
                                        + 1 / 6 * self.sin_3(1) * np.power(x - 1, 3)
                                        + 1 / 24 * self.sin_0(1) * np.power(x - 1, 4),
                              x_range=[-10, 10], use_smoothing=True, color=PURE_RED)

        vg_order_graph = VGroup(order0_plot, order1_plot, order2_plot, order3_plot, order4_plot)

        box = vg_box[0]
        self.play(Create(box))
        order_graph = vg_order_graph[0]
        self.play(FadeTransform(box.copy(), order_graph))

        for i in range(1, len(vg_box)):
            self.play(Transform(box, vg_box[i]))
            self.play(Transform(order_graph, vg_order_graph[i]))
            self.wait(2)

        self.play(FadeOut(box), FadeOut(order_graph))
        self.wait(2)

    def graph_0(self):
        x0_label = self.func_ax[0].get_T_label(x_val=0, graph=self.func_ax[1], label=MathTex(0)).scale(0.9)
        self.play(Transform(self.anchor_label, x0_label))
        self.wait(2)

        sin_tylor = MathTex("\sin(x)=", "\\frac{f(0)}{0 !}", "+",
                            "\\frac{f^{\prime}(0)}{1 !}x", "+",
                            "\\frac{f^{\prime\prime}(0)}{2 !}x^2", "+",
                            "\\frac{f^{\prime\prime\prime}(0)}{3 !}x^3", "+",
                            "\\frac{f^{\prime\prime\prime\prime}(0)}{4 !}x^4", "+",
                            "\ldots", ).scale(0.5).next_to(self.tylor_tex, 2 * DOWN)

        self.play(Transform(self.tylor_example, sin_tylor))
        self.wait(2)

        vg_brace = VGroup(Brace(self.tylor_example[1], DOWN),
                          Brace(self.tylor_example[3], DOWN),
                          Brace(self.tylor_example[5], DOWN),
                          Brace(self.tylor_example[7], DOWN),
                          Brace(self.tylor_example[9], DOWN))

        vg_sin0_text = VGroup(MathTex('\sin(0)=0', color=RED).scale(0.4).next_to(vg_brace[0], DOWN),
                              MathTex('\cos(0)=1', color=GREEN).scale(0.4).next_to(vg_brace[1], DOWN),
                              MathTex('-\sin(0)=0', color=RED).scale(0.4).next_to(vg_brace[2], DOWN),
                              MathTex('-\cos(0)=-1', color=GREEN).scale(0.4).next_to(vg_brace[3], DOWN),
                              MathTex('\sin(0)=0', color=RED).scale(0.4).next_to(vg_brace[4], DOWN))

        for i in range(len(vg_brace)):
            self.play(Create(vg_brace[i]), Write(vg_sin0_text[i]))

        self.wait(3)

        self.play(FadeOut(VGroup(vg_brace[0], vg_brace[2], vg_brace[4])),
                  FadeOut(VGroup(vg_sin0_text[0], vg_sin0_text[2], vg_sin0_text[4])),
                  FadeOut(VGroup(self.tylor_example[1], self.tylor_example[5], self.tylor_example[9])))

        self.wait(2)

        self.play(FadeOut(VGroup(vg_brace[1], vg_brace[3])),
                  FadeOut(VGroup(vg_sin0_text[1], vg_sin0_text[3])))
        self.wait(1)

        sin_tylor_0 = MathTex("\sin(x)=", "x", "-\\frac{x^3}{3 !}",
                              "+\\frac{x^5}{5 !}",
                              "-\\frac{x^7}{7 !}", "+\\frac{x^9}{9 !}",
                              "\ldots", "=",
                              "\sum_{n=0}^{\infty} \\frac{(-1)^n}{(2 n+1) !} x^{2 n+1}").scale(0.5).next_to(
            self.tylor_tex, 2 * DOWN).align_to(self.tylor_example, LEFT)
        self.play(Transform(self.tylor_example, sin_tylor_0))
        self.wait(2)

        self.play(Indicate(sin_tylor_0[-1], run_time=2))
        self.wait(2)

        self.tylor_example = sin_tylor_0

    def graph_0_plot(self):
        vg_box = VGroup(*[SurroundingRectangle(self.tylor_example[i], buff=.05) for i in [1, 2, 3, 4, 5]])
        ax = self.func_ax[0]

        # 零阶
        order0_plot = ax.plot(lambda x: x, x_range=[-10, 10], use_smoothing=True, color=RED)

        # 一阶
        order1_plot = ax.plot(lambda x: x - np.power(x, 3) / 6, x_range=[-10, 10], use_smoothing=True, color=BLUE)

        # 二阶
        order2_plot = ax.plot(lambda x: x - np.power(x, 3) / 6 + np.power(x, 5) / 120,
                              x_range=[-10, 10], use_smoothing=True, color=GREEN)

        # 三阶
        order3_plot = ax.plot(lambda x: x - np.power(x, 3) / 6 + np.power(x, 5) / 120 - np.power(x, 7) / 5040,
                              x_range=[-10, 10], use_smoothing=True, color=MAROON)

        # 四阶
        order4_plot = ax.plot(
            lambda x: x - np.power(x, 3) / 6 + np.power(x, 5) / 120 - np.power(x, 7) / 5040 + np.power(x, 9) / 362880,
            x_range=[-10, 10], use_smoothing=True, color=PURE_RED)

        vg_order_graph = VGroup(order0_plot, order1_plot, order2_plot, order3_plot, order4_plot)

        box = vg_box[0]
        self.play(Create(box))
        order_graph = vg_order_graph[0]
        self.play(FadeTransform(box.copy(), order_graph))

        for i in range(1, len(vg_box)):
            self.play(Transform(box, vg_box[i]))
            self.play(Transform(order_graph, vg_order_graph[i]))
            self.wait(1)

        self.play(FadeOut(box), FadeOut(order_graph))
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


class Taylor_3D(ThreeDScene):
    def construct(self):
        self.ax = VGroup()
        self.tylor_tex = None
        self.tylor_example = None
        self.anchor_label = None

        self.write_twoD_taylor()
        self.update_from_matrix()
        self.get_surfaceplot()
        self.graph_order()

    def write_twoD_taylor(self):
        tylor_2D_0 = MathTex("f(x, y)=", "f(x_0, y_0)").scale(0.6).to_edge(UP + LEFT)
        tylor_2D_1 = MathTex("+", "f_x^{\prime}(x_0, y_0)(x-x_0)",
                             "+f_y^{\prime}(x_0, y_0)(y-y_0)", color=RED).scale(0.55).next_to(tylor_2D_0,
                                                                                              DOWN).align_to(
            tylor_2D_0[1], LEFT)
        tylor_2D_2 = MathTex("+", "\\frac{1}{2 !}[",
                             "f_{x x}^{\prime \prime}(x_0, y_0)(x-x_0)^2",
                             "+f_{x y}^{\prime \prime}(x_0, y_0)(x-x_0)(y-y_0)",
                             "+f_{y x}^{\prime \prime}(x_0, y_0)(x-x_0)(y-y_0)",
                             "+f_{y y}^{\prime \prime}(x_0, y_0)(y-y_0)^2]", color=MAROON).scale(0.5).next_to(
            tylor_2D_1, DOWN).align_to(tylor_2D_1[0], LEFT)
        tylor_2D_3 = MathTex("+\mathcal{O}(x^2,y^2)").scale(0.6).next_to(tylor_2D_2, DOWN).align_to(tylor_2D_2[0], LEFT)

        self.tylor_tex = VGroup(tylor_2D_0, tylor_2D_1, tylor_2D_2, tylor_2D_3)
        self.play(Create(self.tylor_tex))
        self.wait(2)


    def update_from_matrix(self):
        vectorizer = MathTex(r"\begin{bmatrix} x \\ y \end{bmatrix} \Rightarrow \mathbf{\omega}")
        self.play(Write(vectorizer))
        self.wait(2)

        tylor_2D_0 = MathTex("f(\mathbf{\omega})=", "f(\mathbf{\omega_0})").scale(0.7).to_edge(UP + LEFT)
        tylor_2D_1 = MathTex("+", "(\mathbf{\omega} - \mathbf{\omega_0})^T",
                             r"\begin{bmatrix} f_x^{\prime}(\mathbf{\omega_0}) \\ f_y^{\prime}(\mathbf{\omega_0}) \end{bmatrix}",
                             color=RED).scale(0.7).next_to(tylor_2D_0, 0.2*RIGHT)#.align_to(tylor_2D_0[1], LEFT)
        tylor_2D_2 = MathTex("+", "\\frac{1}{2 !}[",
                             "(\mathbf{\omega} - \mathbf{\omega_0})^T",
                             r"\begin{bmatrix} f_{xx}^{\prime \prime}(\mathbf{\omega_0}) & f_{xy}^{\prime \prime}(\mathbf{\omega_0}) \\ f_{yx}^{\prime \prime}(\mathbf{\omega_0}) & f_{yy}^{\prime \prime}(\mathbf{\omega_0}) \end{bmatrix}",
                             "(\mathbf{\omega} - \mathbf{\omega_0})]",
                             color=MAROON).scale(0.7).next_to(tylor_2D_1, 0.2*RIGHT)#.align_to(tylor_2D_1[0], LEFT)
        tylor_2D_3 = MathTex("+\mathcal{O}(\mathbf{\omega}^2)").scale(0.7).next_to(tylor_2D_2, 0.2*RIGHT)#.align_to(tylor_2D_2[0], LEFT)

        self.play(FadeTransform(VGroup(vectorizer.copy(), self.tylor_tex[0]), tylor_2D_0))
        self.play(FadeTransform(VGroup(vectorizer.copy(), self.tylor_tex[1]), tylor_2D_1))
        self.play(FadeTransform(VGroup(vectorizer.copy(), self.tylor_tex[2]), tylor_2D_2))
        self.play(FadeTransform(VGroup(vectorizer.copy(), self.tylor_tex[3]), tylor_2D_3))

        self.wait(2)

        self.tylor_tex = VGroup(tylor_2D_0, tylor_2D_1, tylor_2D_2, tylor_2D_3)

        gradient_brace = Brace(tylor_2D_1[2], DOWN, color=BLUE)

        gradient = MathTex('\mathbf{g}').next_to(gradient_brace, DOWN)

        self.play(Write(gradient), Write(gradient_brace))
        self.wait(2)

        hessian_brace = Brace(tylor_2D_2[3], DOWN, color=BLUE)

        hessian = MathTex('\mathbf{H}').next_to(hessian_brace, DOWN)

        self.play(Write(hessian_brace), Write(hessian))
        self.wait(2)

        self.play(FadeOut(VGroup(tylor_2D_1[2], gradient_brace)),
                  gradient.animate.move_to(tylor_2D_1[2]))

        self.play(FadeOut(VGroup(tylor_2D_2[3], hessian_brace)),
                  hessian.animate.move_to(tylor_2D_2[3]))

        self.play(FadeOut(vectorizer))

    @staticmethod
    def f(x, y):
        z = 1 / 2 * (np.power(x, 2) + np.power(y, 2))
        return np.array([x, y, z])

    def get_surfaceplot(self):
        def bowl(u, v):
            z = 1 / 2 * (np.power(u, 2) + np.power(v, 2))
            return z

        axes = ThreeDAxes(x_range=(-4, 4), y_range=(-4, 4), z_range=(0, 7), x_length=8, y_length=8, z_length=4)
        self.ax.add(axes)
        surface_plane = Surface(lambda u, v: axes.c2p(u, v, bowl(u, v)),
                                u_range=[-2, 2],
                                v_range=[-2, 2],
                                resolution=(30, 30),
                                should_make_jagged=True,
                                stroke_width=0.2,
                                fill_opacity=0.4)

        surface_plane.set_style(fill_opacity=0.4, stroke_color=GRAY)

        surface_plane.set_fill_by_value(axes=axes, colors=[(RED, 0.0), (YELLOW, 0.2), (BLUE, 4)], axis=2)

        #surface_plane.set_fill_by_checkerboard(RED,BLUE,opacity=0.5)

        self.ax.add(surface_plane)

        self.play(Create(self.ax))

        self.play(Rotate(self.ax, angle=-90 * DEGREES, axis=RIGHT))
        self.wait(2)

        for i in range(6):
            self.play(Rotate(self.ax, angle=60 * DEGREES, axis=UP))

        self.wait(2)

        dot_11 = Dot3D(axes.c2p(1, 1, 1), radius=0.5 * DEFAULT_DOT_RADIUS, color=MAROON)

        self.play(Create(dot_11))
        self.wait(2)

        self.ax.add(dot_11)

        # self.play(Rotate(self.ax, angle=15 * DEGREES, axis=UP))
        # self.play(Rotate(self.ax, angle=45 * DEGREES, axis=RIGHT))

    def graph_order(self):
        ax = self.ax[0]

        # 零阶
        order0_plot = Surface(
            lambda u, v: ax.c2p(u, v, self.f_0(1, 1)),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(30, 30),
            should_make_jagged=True,
            stroke_width=0.2,
            fill_opacity=0.6)

        order0_plot.set_style(fill_opacity=0.6, stroke_color=GREEN)
        order0_plot.set_fill_by_checkerboard(BLUE, ORANGE, opacity=0.6)

        # 一阶
        order1_plot = Surface(
            lambda u, v: ax.c2p(u, v, self.f_1(u, v, 1,1)),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(30, 30),
            should_make_jagged=True,
            stroke_width=0.2,
            fill_opacity=0.6)

        order1_plot.set_style(fill_opacity=0.6, stroke_color=GREEN)
        order1_plot.set_fill_by_checkerboard(ORANGE, BLUE, opacity=0.6)

        # 二阶
        order2_plot = Surface(
            lambda u, v: ax.c2p(u, v, self.f_2(u, v, 1, 1)),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(30, 30),
            should_make_jagged=True,
            stroke_width=0.2,
            fill_opacity=0.6)

        order2_plot.set_style(fill_opacity=0.6, stroke_color=GREEN)
        order2_plot.set_fill_by_checkerboard(ORANGE, BLUE, opacity=0.6)

        for i in range(3):
            self.play(Rotate(self.ax, angle=120 * DEGREES, axis=UP))

        self.wait(2)


        vg_order_graph = VGroup(order0_plot, order1_plot, order2_plot)
        vg_box = VGroup(*[SurroundingRectangle(self.tylor_tex[i], buff=.05) for i in [0, 1, 2]])

        box = vg_box[0]
        self.play(Create(box))
        order_graph = vg_order_graph[0]
        self.play(FadeTransform(box.copy(), order_graph))

        self.ax.add(order_graph)
        for i in range(4):
            self.play(Rotate(self.ax, angle=90 * DEGREES, axis=UP))

        for i in range(1, len(vg_box)):
            self.play(Transform(box, vg_box[i]))
            self.play(Transform(order_graph, vg_order_graph[i]))
            for i in range(4):
                self.play(Rotate(self.ax, angle=90 * DEGREES, axis=UP))


        # self.play(FadeOut(box), FadeOut(order_graph))
        # self.wait(2)

    @staticmethod
    def f_0(u, v):
        z = 1 / 2 * (np.power(u, 2) + np.power(v, 2))
        return z

    def f_1(self, u, v, u0, v0):
        zx = (u0 + 1 / 2 * np.power(v0, 2))*(u-u0)
        zy = (v0 + 1 / 2 * np.power(u0, 2))*(v-v0)
        return zx + zy + self.f_0(u0, v0)

    def f_2(self, u, v, u0, v0):
        zxx = (1 + 1 / 2 * np.power(v0, 2))*np.power(u-u0, 2)
        zyy = (1 + 1 / 2 * np.power(u0, 2))*np.power(v-v0, 2)
        zxy = (u0 + v0)*(u-u0)*(v-v0)
        zyx = (u0 + v0)*(u-u0)*(v-v0)
        zx = (u0 + 1 / 2 * np.power(v0, 2))*(u-u0)
        zy = (v0 + 1 / 2 * np.power(u0, 2))*(v-v0)
        return (zxx+zyy+zxy+zyx)/2 + zx + zy + self.f_0(u0, v0)


class Gradient_descent(Scene):
    def construct(self):
        self.func_ax = None
        self.tylor_tex = None

        self.gradient_graph()
        self.gradient_tylor()
        self.formula_deduce()
        self.slope_move()

    def gradient_graph(self):

        ax = CommonFunc.add_axes(x_range=[0.001, 1.1], y_range=[0, 9], x_length=6, y_length=3.5,
                                 axis_config={"include_tip": True, "include_numbers": False}).scale(1.2)

        square_plot = ax.plot(lambda x: -np.log2(x), x_range=[0.001, 1], use_smoothing=True, color=MAROON)

        self.func_ax = VGroup(ax, square_plot)

        self.play(Create(self.func_ax))

        self.wait(2)

        tylor_tex = MathTex("\mathcal{L}(",
                            "\omega",
                            ") =", "\mathcal{L}(", "\omega_0", ")+(",
                            "\omega", "-", "\omega_0", ")^T\mathbf{g}(","\omega_0",")").to_edge(UP)

        self.play(FadeTransform(square_plot.copy(), tylor_tex))
        self.wait(2)

        self.tylor_tex = tylor_tex

    def gradient_tylor(self):
        ax = self.func_ax[0]

        dot_1 = Dot(ax.coords_to_point(0.3, -np.log2(0.3)), color=BLUE)
        dot1_tex = MathTex('\omega_1', color=BLUE).next_to(dot_1, UP)

        dot_2 = Dot(ax.coords_to_point(0.5, -np.log2(0.5)), color=GREEN)
        dot2_tex = MathTex('\omega_2', color=GREEN).next_to(dot_2, UP)

        self.play(Create(dot_1), Write(dot1_tex))

        self.wait(2)

        self.play(Create(dot_2), Write(dot2_tex))

        self.wait(2)

        a, b = dot2_tex.copy(), dot2_tex.copy()

        self.play(FadeOut(self.tylor_tex[1]),
                  FadeOut(self.tylor_tex[6]),
                  a.animate.move_to(self.tylor_tex[1]),
                  b.animate.move_to(self.tylor_tex[6]))

        self.wait(2)

        c, d, e = dot1_tex.copy(), dot1_tex.copy(), dot1_tex.copy()

        self.play(FadeOut(self.tylor_tex[4]),
                  FadeOut(self.tylor_tex[8]),
                  FadeOut(self.tylor_tex[10]),
                  c.animate.move_to(self.tylor_tex[4]),
                  d.animate.move_to(self.tylor_tex[8]),
                  e.animate.move_to(self.tylor_tex[10]))
        self.wait(2)

        self.func_ax.add(VGroup(dot_1, dot1_tex, dot_2, dot2_tex))
        self.tylor_tex.add(VGroup(a,b,c,d,e))

    def formula_deduce(self):
        tylor_tex = MathTex("\mathcal{L}(",
                            "\omega_2",
                            ") =", "\mathcal{L}(", "\omega_1", ")+(",
                            "\omega_2", "-", "\omega_1", ")^T\mathbf{g}(", "\omega_1", ")").to_edge(UP)

        tylor_tex[1].set_color(GREEN)
        tylor_tex[6].set_color(GREEN)
        tylor_tex[4].set_color(BLUE)
        tylor_tex[8].set_color(BLUE)
        tylor_tex[10].set_color(BLUE)

        self.play(self.func_ax.animate.to_edge(RIGHT),
                  Transform(self.tylor_tex, tylor_tex))
        self.wait(2)

        tylor_tex1 = MathTex("\mathcal{L}(",
                            "\omega_2",")-",
                            "\mathcal{L}(", "\omega_1", ")=(",
                            "\omega_2", "-", "\omega_1", ")^T\mathbf{g}(", "\omega_1", ")").scale(0.8).next_to(self.func_ax, LEFT).shift(UP)

        tylor_tex1[1].set_color(GREEN)
        tylor_tex1[6].set_color(GREEN)
        tylor_tex1[4].set_color(BLUE)
        tylor_tex1[8].set_color(BLUE)
        tylor_tex1[10].set_color(BLUE)

        self.play(FadeTransform(tylor_tex.copy(), tylor_tex1))
        self.wait(2)


        self.play(Indicate(self.func_ax[-1]))
        self.wait(1)

        brace_loss = Brace(tylor_tex1[:5], DOWN, color=MAROON)
        tex_loss = MathTex('\leq 0').next_to(brace_loss, DOWN)

        self.play(FadeIn(brace_loss), Create(tex_loss))
        self.wait(2)

        g_square = MathTex("\mathbf{g}(\omega_1)^T",
                           "\mathbf{g}(", "\omega_1", ")\geq 0").scale(0.8).next_to(self.func_ax, LEFT).shift(DOWN).align_to(tylor_tex1[6], LEFT)
        self.play(FadeTransform(tylor_tex1[9:].copy(), g_square))
        self.wait(2)

        g_square2 = MathTex("-\mathbf{g}(\omega_1)^T",
                           "\mathbf{g}(", "\omega_1", ")\leq 0").scale(0.8).next_to(self.func_ax, LEFT).shift(DOWN).align_to(tylor_tex1[5], LEFT)

        self.play(Transform(g_square, g_square2))
        self.wait(2)

        g_square3 = MathTex("-\epsilon \mathbf{g}(\omega_1)^T",
                           "\mathbf{g}(", "\omega_1", ")\leq 0").scale(0.8).next_to(self.func_ax, LEFT).shift(DOWN).align_to(tylor_tex1[5], LEFT)
        g_square_epsilon = MathTex('\epsilon > 0').scale(0.9).next_to(g_square3, DOWN)

        self.play(Transform(g_square, g_square3))
        self.play(FadeIn(g_square_epsilon))
        self.wait(2)

        #copy_diff = tylor_tex1[6:9].copy().shift(DOWN)

        brace_diff = Brace(tylor_tex1[6:9], DOWN, color=RED)
        self.play(Create(brace_diff),
                  g_square[0].copy().animate.move_to(brace_diff).shift(0.4*DOWN))
        self.wait(2)

        gradient_descent = MathTex("\omega_2", "=", "\omega_1-",
                                   "\epsilon","\mathbf{g}(\omega_1)").to_edge(RIGHT).shift(UP)

        self.play(FadeTransform(VGroup(tylor_tex1[6:9].copy(), g_square.copy()), gradient_descent))
        self.wait(2)

        self.play(Indicate(gradient_descent[3]))

    def slope_move(self):

        ax, graph = self.func_ax[0], self.func_ax[1]

        slope = ax.get_secant_slope_group(
            x=0.3,
            graph=graph,
            dx=0.001,
            dx_line_color=YELLOW,
            secant_line_length=3,
            secant_line_color=YELLOW,
        )

        vg_slopes = VGroup(*[ax.get_secant_slope_group(x=x,
                                                       graph=graph, dx=0.01,
                                                       dx_line_color=YELLOW,
                                                       secant_line_length=3,
                                                       secant_line_color=YELLOW) for x in np.linspace(0.3, 0.5, 10)])

        self.play(Create(slope))
        self.wait(2)

        self.play(Succession(*[Transform(slope, vg_slopes[i]) for i in range(len(vg_slopes))]))

        self.wait(2)


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

class Next_Epsiode(ThreeDScene):
    def construct(self):
        self.text = None
        self.ax = VGroup()
        self.m = 10
        self.loss_track = None
        self.loss_graph = None
        self.iterations = None
        self.learning_rate = [0.5, 2, 2.05]
        self.lr_text = None

        title = Text("优化算法下一期").to_edge(UP)
        self.play(FadeIn(title))

        self.get_surfaceplot()

    @staticmethod
    def f(x, y):
        z = 1 / 2 * (np.power(x, 2) + np.power(y, 2))
        return np.array([x, y, z])

    def get_surfaceplot(self):
        def bowl(u, v):
            z = 1 / 2 * (np.power(u, 2) - np.power(v, 2))
            return z

        axes = ThreeDAxes(x_range=(-4, 4), y_range=(-4, 4), z_range=(0, 7), x_length=8, y_length=8, z_length=4)
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


class Opt_VF(Scene):
    def construct(self):
        func = lambda pos: pos + LEFT * 2
        colors = [RED, YELLOW, BLUE, DARK_GRAY]
        # min_radius = Circle(radius=2, color=colors[0]).shift(LEFT * 5)
        # max_radius = Circle(radius=10, color=colors[-1]).shift(LEFT * 5)
        vf = ArrowVectorField(func, min_color_scheme_value=2, max_color_scheme_value=10, colors=colors).scale(0.4)
        self.play(Write(vf))
        self.wait(2)
        # self.add(vf, min_radius, max_radius)


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
                                "\epsilon","\mathbf{g}(\omega_m)").to_edge(LEFT + UP)
        text_gradient[2].set_color(MAROON)

        g_text = MathTex('\mathbf{g} = \\nabla \mathcal{L}').scale(0.6).next_to(text_gradient, DOWN)

        self.play(Write(text_gradient), Write(g_text))
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


class thanks_end(Scene):
    def construct(self):
        svg_image = SVGMobject('svg_icon/bird.svg', fill_color=MAROON).scale(1.2).to_edge(6 * LEFT + UP)

        text = Text('感谢充电', font='SIL-Hei-Med-Jian').scale(1.2).next_to(svg_image, 2.5 * DOWN)

        self.play(SpinInFromNothing(svg_image))

        self.play(Create(text))

        source_path = 'svg_icon/charge/'

        l_image_path = [image_path for image_path in os.listdir(source_path) if image_path.split('.')[-1] == 'jpg']
        vg_anchor = VGroup(*[Circle(1) for _ in range(len(l_image_path))])
        vg_anchor.arrange_in_grid(rows=3, buff=0.1).to_edge(2 * RIGHT)
        # self.add(vg_anchor)

        for i in range(len(l_image_path)):
            image_path = l_image_path[i]
            split_str = image_path.split('.')
            if split_str[-1] == 'jpg':
                name = split_str[0]
                # image = cv2.imread(os.path.join(source_path, image_path),3)
                # resize_array = cv2.resize(image, (512,512))
                image = ImageMobject(os.path.join(source_path, image_path)).move_to(vg_anchor[i].get_center())
                image.height = 1.2
                image.width = 1.2
                name_text = Text(name).scale(0.55).next_to(image, 0.5 * DOWN)
                self.add(image, name_text)

        self.wait(3)
