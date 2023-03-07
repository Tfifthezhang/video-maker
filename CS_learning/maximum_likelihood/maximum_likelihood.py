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

np.random.seed(0)


class NormalDistribution(Scene):
    def construct(self):
        self.axes = None
        self.sigma = 1
        self.mu = 0
        self.formula = None
        self.graph = None
        self.var_tracker = VGroup()
        self.sampler = None

        self.write_gaussian()
        self.update_var()
        self.sampling_from_normal()
        # self.render_diagram()

    def write_gaussian(self):
        math_formula = MathTex(
            'f(x)=\\frac{1}{\sigma \sqrt{2 \pi}} e^{-\\frac{1}{2}\left(\\frac{x-\mu}{\sigma}\\right)^2}').to_edge(
            UP + LEFT)

        normal_formula = MathTex("\mathcal{N}", "(", "\mu", ",", "\sigma", ")").move_to(math_formula.get_center())

        self.play(Write(math_formula))
        self.wait(1)
        self.play(ReplacementTransform(math_formula, normal_formula))
        self.wait(2)
        self.play(normal_formula[2].animate.set_color(BLUE))
        self.play(normal_formula[4].animate.set_color(RED))

        self.formula = normal_formula

    def update_var(self):
        mu = CommonFunc.variable_tracker(label=Tex('$\mu$'), color=BLUE, var_type=DecimalNumber).scale(0.8).to_edge(
            LEFT)
        self.var_tracker.add(mu)
        sigma = CommonFunc.variable_tracker(label=Tex('$\sigma$'), color=RED, var_type=DecimalNumber).scale(
            0.9).next_to(mu, DOWN)
        self.var_tracker.add(sigma)
        ax = CommonFunc.add_axes(x_range=[-6, 6], y_range=[0, 1], x_length=8, y_length=6,
                                 axis_config={"include_tip": True, "include_numbers": False}).scale(0.8).shift(UP)
        self.axes = ax
        self.play(Create(ax))

        graph = ax.plot(lambda x: self.normal_dis(x, mu=self.mu, sigma=self.sigma), x_range=[self.mu - 3, self.mu + 3],
                        use_smoothing=True)
        self.graph = graph
        self.play(Create(graph))

        self.play(Create(mu), Create(sigma))

        for i in np.linspace(-4, 4, 15).tolist():
            self.play(mu.tracker.animate.set_value(i))
            self.play(sigma.tracker.animate.set_value(self.sigma))
            graph_other = ax.plot(lambda x: self.normal_dis(x, mu=i, sigma=self.sigma), x_range=[i - 3, i + 3],
                                  use_smoothing=True, color=MAROON)
            self.play(Transform(graph, graph_other))

        self.wait(2)

        for j in np.linspace(0.5, 2.5, 15).tolist():
            self.play(mu.tracker.animate.set_value(self.mu))
            self.play(sigma.tracker.animate.set_value(j))
            graph_other = ax.plot(lambda x: self.normal_dis(x, mu=self.mu, sigma=j), x_range=[-6, 6],
                                  use_smoothing=True, color=MAROON)
            self.play(Transform(graph, graph_other))

        self.play(FadeOut(graph))

        self.wait(1)

        self.play(mu.tracker.animate.set_value(self.mu))
        self.play(sigma.tracker.animate.set_value(self.sigma))
        graph = ax.plot(lambda x: self.normal_dis(x, mu=self.mu, sigma=self.sigma), x_range=[self.mu - 3, self.mu + 3],
                        use_smoothing=True)
        self.graph = graph
        self.play(FadeIn(self.graph))

    def normal_dis(self, x, sigma, mu):
        coef = 1 / (sigma * np.sqrt(2 * np.pi))
        expon = -1 / 2 * ((x - mu) / sigma) ** 2
        return coef * np.power(np.e, expon)

    def sampling_from_normal(self):
        sampler = np.random.normal(loc=0, scale=1, size=10)

        vg_sample = VGroup(*[DecimalNumber(n) for n in sampler])

        vg_sample.arrange_submobjects(DOWN, buff=0.25).scale(0.8).to_edge(2 * RIGHT)
        self.sampler = vg_sample

        tracker = ValueTracker(0)
        pointer = Vector(UP).next_to(self.axes.c2p(tracker.get_value()), DOWN)
        label = Text('x').add_updater(lambda m: m.next_to(pointer, RIGHT))
        # label = CommonFunc.variable_tracker(label=MathTex('x'), var_type=DecimalNumber).scale(0.8)
        label.add_updater(lambda m: m.next_to(pointer, RIGHT))
        pointer.add_updater(lambda m: m.next_to(self.axes.c2p(tracker.get_value()), DOWN))

        self.play(Create(pointer), Create(label))
        for i in range(len(sampler)):
            self.play(tracker.animate.set_value(sampler[i]))
            copy_label = label.copy()
            self.play(FadeTransform(copy_label, vg_sample[i], stretch=True))

        self.play(Uncreate(pointer), Uncreate(label))

    def key_question(self):
        self.play(Uncreate(self.axes),
                  Uncreate(self.graph),
                  Uncreate(self.var_tracker))

        self.play(self.formula.animate.move_to(LEFT))
        self.play(self.sampler.animate.move_to(RIGHT))


class MaxProbability(ThreeDScene):
    def construct(self):
        self.sample_array = None
        self.vg_sample = None
        self.vg_graph = None
        self.vg_formula = None
        self.prod_formula = None

        self.mle_sample()
        self.mle_normal()
        self.mle_explain()
        self.mle_prod()
        # self.mle_3D()

    def mle_sample(self):
        sampler = np.random.normal(loc=1, scale=0.5, size=10)

        self.sample_array = sampler
        vg_sample = VGroup(*[DecimalNumber(n) for n in sampler])
        vg_sample.arrange_submobjects(DOWN, buff=0.3).scale(0.8).to_edge(2 * LEFT)

        self.vg_sample = vg_sample

        self.play(Create(vg_sample))

        self.wait(1)

    def mle_normal(self):
        vg_graph = VGroup()
        l_mu = np.random.uniform(-3, 3, 8)
        l_sigma = np.random.uniform(0.5, 3, 8)
        for i, j in list(zip(l_mu, l_sigma)):
            ax = CommonFunc.add_axes(x_range=[-10, 10], y_range=[0, 0.7], x_length=8, y_length=6,
                                     axis_config={"include_tip": True, "include_numbers": False}).scale(0.6)
            graph = ax.plot(lambda x: self.normal_dis(x, mu=i, sigma=j), x_range=[i - 10, i + 10],
                            use_smoothing=True)
            graph_label = ax.get_graph_label(graph=graph,
                                             label=MathTex('\mu={:.2f},\sigma^2={:.2f}'.format(i, j)),
                                             direction=DL).scale(0.8)

            ax_vg = VGroup(ax, graph, graph_label)
            vg_graph.add(ax_vg)

        vg_graph.arrange_in_grid(2, 4, buff=0.15).scale(0.5).next_to(self.vg_sample, 2 * RIGHT)

        self.vg_graph = vg_graph

        self.play(Write(vg_graph))

        self.wait(3)

        self.play(Unwrite(self.vg_graph))

    def mle_explain(self):

        vg_formula = VGroup(
            *[MathTex("\mathcal{N}", "(", "{:.2f}".format(i), "|", "\mu", ",", "\sigma", ")").scale(0.7) for i in
              self.sample_array])

        vg_formula.arrange_submobjects(DOWN, buff=0.3).next_to(self.vg_sample, 3 * RIGHT)

        for i in range(len(self.vg_sample)):
            copy_sample = self.vg_sample[i].copy()
            self.play(FadeTransform(copy_sample, vg_formula[i], stretch=True))

        self.wait(3)

        brace = Brace(vg_formula, RIGHT, buff=0.2)

        prod_formula = MathTex("\prod_i^n", "\mathcal{N}", "(", "x_i", "|", "\mu", ",", "\sigma", ")").scale(
            0.8).next_to(brace, RIGHT)

        self.play(Create(brace, run_time=2))
        self.wait(1)
        self.play(Write(prod_formula))

        mle_formula = MathTex("\max", "[", "\prod_i^n", "\mathcal{N}", "(", "x_i", "|", "\mu", ",", "\sigma", ")",
                              "]").scale(0.8)

        self.play(ReplacementTransform(prod_formula, mle_formula))

        self.play(mle_formula[0].animate.set_color(MAROON))
        self.play(mle_formula[7].animate.set_color(BLUE))
        self.play(mle_formula[9].animate.set_color(RED))

        self.wait(2)

        self.prod_formula = mle_formula

    def mle_prod(self):
        # 由于多个小的概率连乘会造成下溢,所以我们想把乘法干掉
        self.play(Indicate(self.prod_formula[2], run_time=3))

        ax = CommonFunc.add_axes(x_range=[0, 1.1], y_range=[0, 1.3], x_length=6, y_length=6,
                                 axis_config={"include_tip": True, "include_numbers": True}).scale(0.6).next_to(
            self.prod_formula, RIGHT)
        graph_prod = ax.plot(lambda x: x ** 2, x_range=[0, 1.1], use_smoothing=True, color=BLUE)
        graph_linear = ax.plot(lambda x: x, x_range=[0, 1.1], use_smoothing=True, color=YELLOW)
        graph_label_prod = ax.get_graph_label(graph=graph_prod,
                                              label=MathTex('x \\times x')).scale(0.7)
        graph_label_linear = ax.get_graph_label(graph=graph_linear,
                                                label=MathTex('y=x')).scale(0.7).next_to(graph_prod, UP)
        self.play(Create(ax))
        self.play(Create(graph_prod), Write(graph_label_prod),
                  Create(graph_linear), Write(graph_label_linear))

        area = ax.get_area(graph_linear, [0.01, 1], bounded_graph=graph_prod, color=RED, opacity=0.5)

        self.play(Create(area, run_time=3))

        self.wait(2)

        self.play(Uncreate(ax), Uncreate(graph_prod), Uncreate(graph_label_prod),
                  Uncreate(graph_linear), Uncreate(graph_label_linear), Uncreate(area))

        # 所以我们不再寻找参数使得联合概率最大，而是让它的对数最大

        ln_formula = MathTex("\max", "\ln", "[", "\prod_i^n", "\mathcal{N}", "(", "x_i", "|", "\mu", ",", "\sigma", ")",
                             "]").scale(0.8)
        self.play(ReplacementTransform(self.prod_formula, ln_formula))
        self.play(Indicate(self.prod_formula[1], run_time=2))

        brace_ln = Brace(ln_formula[1:], DOWN)
        brace_text = MathTex("\sum_i^n", "\ln", "\mathcal{N}", "(", "x_i", "|", "\mu", ",", "\sigma", ")").next_to(
            brace_ln, DOWN)
        self.play(Write(brace_ln), Write(brace_text))

        self.wait(2)

        final_formula = MathTex("\max", "\sum_i^n", "\ln", "\mathcal{N}", "(", "x_i", "|", "\mu", ",", "\sigma",
                                ")").scale(0.8)

        self.play(FadeOut(brace_ln), FadeOut(self.prod_formula), FadeOut(ln_formula))
        self.play(FadeTransform(brace_text, final_formula))

        # 注意到最终的式子中是求最大值，所以不能改变其单调性

        self.play(Indicate(final_formula[0], run_time=2))

        bx = CommonFunc.add_axes(x_range=[0.001, 6], y_range=[-8, 2], x_length=6, y_length=4,
                                 axis_config={"include_tip": True, "include_numbers": True}).scale(0.7).next_to(
            final_formula, RIGHT)
        graph_ln = bx.plot(lambda x: np.log(x), x_range=[0.001, 6], use_smoothing=True, color=MAROON)
        graph_label_ln = bx.get_graph_label(graph=graph_ln, label=MathTex('\\ln(x)')).scale(0.8).next_to(graph_ln, DOWN)
        self.play(Create(bx))
        self.play(Create(graph_ln), Write(graph_label_ln))

        self.wait(3)

        self.prod_formula = final_formula

        # self.play(Uncreate(bx), Uncreate(graph_ln), Uncreate(graph_label_ln))

    def normal_dis(self, x, sigma, mu):
        coef = 1 / (sigma * np.sqrt(2 * np.pi))
        expon = -1 / 2 * ((x - mu) / sigma) ** 2
        return coef * np.power(np.e, expon)


class MLE_3D(ThreeDScene):
    def construct(self):
        self.sampler = np.random.normal(loc=1, scale=0.5, size=10)
        self.mle_3D()

    def func_mle(self, mu, sigma):
        s = 0
        for x in self.sampler:
            weight = 0.5 * np.power((x - mu) / sigma, 2)
            s += weight
        return s / 20000

    def mle_3D(self):

        axes = ThreeDAxes(x_range=(-4, 4), y_range=(-4, 4), z_range=(0, 8))
        surface_plane = Surface(lambda u, v: axes.c2p(u, v, self.func_mle(u, v)),
                                u_range=[-3, 3],
                                v_range=[0.1, 3.1],
                                resolution=(30, 30),
                                should_make_jagged=True,
                                stroke_width=0.2)

        surface_plane.set_style(fill_opacity=0.5, stroke_color=RED)

        surface_plane.set_fill_by_value(axes=axes, colors=[(RED, 1), (YELLOW, 2), (BLUE, 4)], axis=2)

        self.add(axes)
        self.play(Create(surface_plane))

        self.move_camera(phi=75 * DEGREES)

        # self.set_camera_orientation(phi=75 * DEGREES, theta=0)
        #
        for i in range(0, 120, 30):
            self.move_camera(theta=i * DEGREES)


class Regression(Scene):
    def construct(self):
        self.dot_linspace = None
        self.dots = None
        self.ax = None
        self.model_formula = None
        self.linear_graph = None

        self.plot_scatter()
        self.regression_classic()
        self.regression_loss()

    def plot_scatter(self):
        ax = CommonFunc.add_axes(x_range=[-8, 8], y_range=[-8, 8], x_length=8, y_length=6,
                                 axis_config={"include_tip": False, "include_numbers": False})
        self.play(Create(ax))
        self.ax = ax

        x = np.linspace(-7.5, 7.5, 150)
        self.dot_linspace = x
        gaussian_noise = np.random.normal(loc=0, scale=3, size=(150,))
        y = 0.8 * x - 0.7
        y_noise = y + gaussian_noise
        coords = list(zip(x, y_noise))

        dots = VGroup(
            *[Dot(ax.c2p(coord[0], coord[1]), radius=0.5 * DEFAULT_DOT_RADIUS, color=BLUE) for coord in coords])
        self.play(FadeIn(dots))

        self.dots = dots

        self.wait(3)

    def regression_classic(self):
        # 考虑线性回归模型
        weight_formula = MathTex('y = \omega x + b').to_edge(UP + LEFT)
        self.play(Write(weight_formula))
        self.wait(2)

        ## 向量形式
        vector_formula = MathTex('\\vec {y} = \\vec{\omega}^{\\top} X').to_edge(UP + LEFT)
        self.play(ReplacementTransform(weight_formula, vector_formula))

        self.wait(3)
        self.model_formula = vector_formula

        inital_plot = self.ax.plot(lambda x: x, x_range=[-7, 7], use_smoothing=True, color=MAROON)
        inital_label = self.ax.get_graph_label(graph=inital_plot, label=MathTex('y=x')).scale(0.8)

        self.play(Write(inital_plot), Write(inital_label))

        l_omega = np.linspace(1.4, 0.5, 20)
        l_b = np.linspace(1.5, 0.1, 20)
        for i, j in zip(l_omega, l_b):
            next_plot = self.ax.plot(lambda x: i * x + j, x_range=[-7, 7], use_smoothing=True, color=MAROON)
            next_label = self.ax.get_graph_label(graph=inital_plot,
                                                 label=MathTex('y={0:.2f}x+{1:.2f}'.format(i, j))).scale(0.8)
            self.play(Transform(inital_plot, next_plot), Transform(inital_label, next_label))

        self.linear_graph = next_plot

    def regression_loss(self):
        # loss_formula = MathTex('\mathcal{L} = (y-\hat{y})^2').next_to(self.model_formula, DOWN)
        # self.play(Write(loss_formula))

        # 那么哪一组参数最合适呢？
        dot = self.dots[100]

        lines_1 = self.ax.get_lines_to_point(dot.get_center(), color=GREEN)

        graph_dot = self.ax.input_to_graph_point(self.dot_linspace[100], graph=self.linear_graph)
        lines_2 = self.ax.get_lines_to_point(graph_dot, color=GREEN)

        self.play(Create(lines_1), Create(lines_2))

        diff_line = CommonFunc.add_line(dot, graph_dot, color=YELLOW)
        diff_math = MathTex('(y-\hat{y})^2').scale(0.7).next_to(diff_line, RIGHT)

        self.play(Write(diff_line))
        self.play(Write(diff_math))

        vg_diff_line = VGroup()
        ## 对每个数据点都进行如此操作
        for i in range(len(self.dots)):
            dot = self.dots[i]
            graph_dot = self.ax.input_to_graph_point(self.dot_linspace[i], graph=self.linear_graph)
            vg_diff_line.add(CommonFunc.add_line(dot, graph_dot, color=YELLOW))

        self.play(Create(vg_diff_line, lag_ratio=0.5, run_time=5))

        self.wait(3)

        # 最后我们得到了损失函数
        loss_formula = MathTex('\mathcal{L} = \sum_i^n (y_i-\hat{y}_i)^2').next_to(self.model_formula, DOWN).shift(
            RIGHT)
        self.play(FadeTransform(vg_diff_line, loss_formula))

        self.wait(3)

        loss_formula2 = MathTex('\min_\omega \sum_i^n (y_i-\omega_ix_i)^2').next_to(self.model_formula, DOWN).shift(
            RIGHT)
        self.play(ReplacementTransform(loss_formula, loss_formula2))

        self.wait(1)


class MLE_Regression(MovingCameraScene):
    def construct(self):
        self.dot_linspace = None
        self.dots = None
        self.ax = None
        self.model_formula = None
        self.linear_graph = None

        self.plot_scatter()
        self.single_dot_dis()

    def plot_scatter(self):
        ax = CommonFunc.add_axes(x_range=[-8, 8], y_range=[-8, 8], x_length=8, y_length=6,
                                 axis_config={"include_tip": False, "include_numbers": False})
        self.play(Create(ax))
        self.ax = ax

        x = np.linspace(-7.5, 7.5, 150)
        self.dot_linspace = x
        gaussian_noise = np.random.normal(loc=0, scale=3, size=(150,))
        y = 0.8 * x - 0.7
        y_noise = y + gaussian_noise
        coords = list(zip(x, y_noise))

        dots = VGroup(
            *[Dot(ax.c2p(coord[0], coord[1]), radius=0.5 * DEFAULT_DOT_RADIUS, color=BLUE) for coord in coords])
        self.play(FadeIn(dots))

        self.dots = dots

        self.wait(1)

    def single_dot_dis(self):
        self.camera.frame.save_state()

        dot = self.dots[100]
        self.play(self.camera.frame.animate.scale(0.3).move_to(dot))

        self.play(dot.animate.set(color=RED))

        iid_axes = Axes(x_range=[-3, 3], y_range=[0, 0.6], x_length=5, y_length=1,
                        axis_config=dict(include_tip=False,
                                         include_numbers=False,
                                         rotation=0 * DEGREES,
                                         stroke_width=1.0), ).scale(0.3).rotate(270 * DEGREES).next_to(dot,
                                                                                                       0.05 * RIGHT)
        self.play(Create(iid_axes))
        graph = iid_axes.plot(lambda x: self.normal_dis(x, mu=0, sigma=1),
                              x_range=[-3, 3],
                              use_smoothing=True,
                              color=RED)

        self.play(Create(graph))

        graph_tex = MathTex("\mathcal{N}", "(", "y_i", "|", "\omega x_i", ",", "\sigma^2", ")").scale(0.3).next_to(
            graph, RIGHT)

        self.play(Write(graph_tex, run_time=2))

        self.wait(3)
        self.play(Restore(self.camera.frame))

        vg_dis = VGroup()
        ## 对每个数据点都进行如此操作
        for i in range(0, len(self.dots), 5):
            dot = self.dots[i]
            dis_iid_axes = Axes(x_range=[-3, 3], y_range=[0, 0.6], x_length=5, y_length=1,
                            axis_config=dict(include_tip=False,
                                             include_numbers=False,
                                             rotation=0 * DEGREES,
                                             stroke_width=1.0), ).scale(0.3).rotate(270 * DEGREES).next_to(dot,
                                                                                                           0.05 * RIGHT)
            dis_graph = dis_iid_axes.plot(lambda x: self.normal_dis(x, mu=0, sigma=1),
                                  x_range=[-3, 3],
                                  use_smoothing=True,
                                  color=RED)
            vg_dis.add(VGroup(dis_iid_axes, dis_graph))
        self.play(Create(vg_dis, lag_ratio=0.5, run_time=5))

        self.wait(3)



    def normal_dis(self, x, sigma, mu):
        coef = 1 / (sigma * np.sqrt(2 * np.pi))
        expon = -1 / 2 * ((x - mu) / sigma) ** 2
        return coef * np.power(np.e, expon)
