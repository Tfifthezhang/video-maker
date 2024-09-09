# -*- coding: utf-8 -*-
from datetime import datetime
import random
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
        self.vg_svg = None
        self.vg_date = None
        self.tex = None

        self.intro_problem()
        self.intro_calendar()
        #self.intro_combine()
        self.intro_prob()
        #self.data_prepare()
        #self.mse_dot()

    @staticmethod
    def birth_func(k, n):
        up = k*(k-1)
        down = 2*n
        return 1-np.exp(-up/down)

    def random_birthday(self):
        b = datetime(2024, random.randint(1, 12), random.randint(1, 29))
        s_date = b.strftime("%m %d")
        return s_date

    def intro_problem(self):
        n_svg = 24
        vg_svg = VGroup(*[SVGMobject('svg_icon/people.svg', fill_color=WHITE).scale(0.5) for _ in range(n_svg)])

        vg_svg.arrange_in_grid(rows=4, buff=0.6)

        self.play(Create(vg_svg))
        self.wait(1)
        vg_date = VGroup(*[Text(self.random_birthday(),color=YELLOW_C).scale(0.5) for _ in range(n_svg)])
        for n in range(n_svg):
            vg_date[n].next_to(vg_svg[n], 0.5*UP)

        self.play(FadeIn(vg_date))

        self.wait(2)

        for example in range(10):
            new_vg_date = VGroup(*[Text(self.random_birthday(),color=YELLOW_C).scale(0.5) for _ in range(n_svg)])
            for n in range(n_svg):
                new_vg_date[n].next_to(vg_svg[n], 0.5*UP)
            self.play(Transform(vg_date, new_vg_date))
            self.wait()

        self.vg_date = vg_date
        self.vg_svg = vg_svg

    def intro_calendar(self):
        self.play(VGroup(self.vg_date, self.vg_svg).animate.to_edge(LEFT))

        n_circles = 365
        vg_circle = VGroup(*[Circle(radius=0.3, color=BLUE).scale(0.6) for _ in range(n_circles)])
        vg_circle.arrange_in_grid(cols=12, buff=0.1).scale(0.4).to_edge(RIGHT)

        brace_in = Brace(vg_circle, direction=LEFT, color=MAROON)
        text_day = Tex('365').scale(1).next_to(brace_in, 0.4*LEFT)

        self.play(Create(vg_circle))
        self.wait(1)

        self.play(Create(brace_in),
                  Write(text_day))
        self.wait(2)

    def intro_combine(self):
        self.play(FadeOut(self.vg_date))
        self.play(self.vg_svg.animate.scale(0.8).to_edge(UP))

        tex1 = MathTex("P(x=1)").scale(1).next_to(self.vg_svg, DOWN)
        self.play(FadeIn(tex1))

        vg_tex = VGroup()
        for i in range(2, 24):
            tex = MathTex("P(x={})".format(i))
            vg_tex.add(tex)

        vg_tex.arrange_in_grid(rows=3, buff=0.1).scale(0.7).next_to(tex1, 2*DOWN)

        self.play(Write(vg_tex))

        self.play(Indicate(vg_tex[0]))
        self.play(self.vg_svg[12].animate.set_color(RED),
                  self.vg_svg[0].animate.set_color(RED))
        self.wait(1)

        self.play(Indicate(vg_tex[1]))
        self.play(self.vg_svg[15].animate.set_color(RED))
        self.wait(1)

        self.play(Indicate(vg_tex[-1]))
        self.play(self.vg_svg.animate.set_color(RED))
        self.wait(1)

        self.play(Circumscribe(vg_tex))
        self.wait(1)

        tex2 = MathTex("P(x\geq2)").scale(1).next_to(tex1, 2*DOWN)
        self.play(ReplacementTransform(vg_tex, tex2))

        self.wait(1)

        tex_plus = MathTex("+").scale(1).next_to(tex1, DOWN).shift(0.2*UP)
        self.play(FadeIn(tex_plus))

        vg_plus = VGroup(tex1,tex2,tex_plus)

        tex_eq = MathTex(" = 1").scale(1).next_to(vg_plus, RIGHT)
        vg_plus.add(tex_eq)
        self.play(Create(tex_eq))
        self.wait(1)

        tex_final = MathTex("P(x\geq2)", "=", "1", "-", "P(x=1)").next_to(self.vg_svg, DOWN)
        self.play(TransformMatchingTex(vg_plus, tex_final))
        self.wait(1)

        self.tex = tex_final

    def intro_prob(self):
        self.play(FadeOut(self.vg_date))
        vg_prob = VGroup(*[MathTex("\\frac{365-%d}{365}" % i, color=YELLOW).scale(0.45) for i in range(24)])
        for n in range(24):
            vg_prob[n].next_to(self.vg_svg[n], 0.5*UP)

        self.play(FadeIn(vg_prob))
        #self.play()


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


