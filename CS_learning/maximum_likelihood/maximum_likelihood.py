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

        self.mle_sample()
        self.mle_normal()
        self.mle_explain()
        #self.mle_3D()

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

        vg_graph.arrange_in_grid(2, 4, buff=0.15).scale(0.5).next_to(self.vg_sample, 2*RIGHT)

        self.vg_graph = vg_graph

        self.play(Write(vg_graph))

        self.wait(3)

        self.play(Unwrite(self.vg_graph))

    def mle_explain(self):

        vg_formula = VGroup(*[MathTex("\mathcal{N}", "(", "{:.2f}".format(i),"|", "\mu", ",", "\sigma", ")").scale(0.7) for i in self.sample_array])

        vg_formula.arrange_submobjects(DOWN, buff=0.3).next_to(self.vg_sample, 3*RIGHT)

        for i in range(len(self.vg_sample)):
            copy_sample = self.vg_sample[i].copy()
            self.play(FadeTransform(copy_sample, vg_formula[i], stretch=True))

        self.wait(3)


        #self.play(vg_formula.animate.arrange_submobjects(RIGHT, buff=SMALL_BUFF).next_to(self.vg_sample, RIGHT))



    def mle_3D(self):

        def bowl(u, v):
            z = 1 / 2 * (np.power(u, 2) + np.power(v, 2)/5)
            return z

        axes = ThreeDAxes(x_range=(-4, 4), y_range=(-2, 2), z_range=(0, 8))
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

        self.move_camera(phi=75 * DEGREES)

        # self.set_camera_orientation(phi=75 * DEGREES, theta=0)

        # for i in range(0, 120, 30):
        #     self.move_camera(theta=i * DEGREES)










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
                                         stroke_width=1.0), ).scale(0.3).rotate(270 * DEGREES).next_to(dots[140],
                                                                                                       0.05 * RIGHT)
        self.play(Create(iid_axes))

        # self.play(iid_axes.animate.rotate(270*DEGREES))
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
