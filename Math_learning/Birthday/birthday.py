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
        self.poly_ax = VGroup()
        self.linear_formula = None
        self.degree = None
        self.vg_svg = None
        self.vg_date = None
        self.vg_calendar = None
        self.tex = None

        self.intro_problem()
        self.intro_calendar()
        self.intro_combine()
        self.intro_prob()
        # self.intro_prob()
        # self.data_prepare()
        # self.mse_dot()

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
        vg_date = VGroup(*[Text(self.random_birthday(), color=YELLOW_C).scale(0.5) for _ in range(n_svg)])
        for n in range(n_svg):
            vg_date[n].next_to(vg_svg[n], 0.5 * UP)

        self.play(FadeIn(vg_date))

        self.wait(2)

        for example in range(10):
            new_vg_date = VGroup(*[Text(self.random_birthday(), color=YELLOW_C).scale(0.5) for _ in range(n_svg)])
            for n in range(n_svg):
                new_vg_date[n].next_to(vg_svg[n], 0.5 * UP)
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
        text_day = Tex('365').scale(1).next_to(brace_in, 0.4 * LEFT)

        self.play(Create(vg_circle))
        self.wait(1)

        self.play(Create(brace_in),
                  Write(text_day))
        self.wait(2)

        self.vg_calendar = VGroup(vg_circle, brace_in, text_day)

    def intro_combine(self):
        self.play(FadeOut(self.vg_date),
                  FadeOut(self.vg_calendar),
                  FadeOut(self.vg_svg))
        vg_svg_copy = self.vg_svg.copy().move_to(ORIGIN).scale(0.8).to_edge(UP)
        self.play(FadeIn(vg_svg_copy))
        self.wait(1)

        tex1 = MathTex("P(x=1)").scale(1).next_to(vg_svg_copy, DOWN)
        self.play(FadeIn(tex1))

        vg_tex = VGroup()
        for i in range(2, 24):
            tex = MathTex("P(x={})".format(i))
            vg_tex.add(tex)

        vg_tex.arrange_in_grid(rows=3, buff=0.1).scale(0.7).next_to(tex1, 2 * DOWN)

        self.play(Write(vg_tex))

        self.play(Indicate(vg_tex[0]))
        self.play(vg_svg_copy[12].animate.set_color(RED),
                  vg_svg_copy[0].animate.set_color(RED))
        self.wait(1)

        self.play(Indicate(vg_tex[1]))
        self.play(vg_svg_copy[15].animate.set_color(RED))
        self.wait(1)

        self.play(Indicate(vg_tex[-1]))
        self.play(vg_svg_copy.animate.set_color(RED))
        self.wait(1)

        self.play(Circumscribe(vg_tex))
        self.wait(1)

        tex2 = MathTex("P(x\geq2)").scale(1).next_to(tex1, 2 * DOWN)
        self.play(ReplacementTransform(vg_tex, tex2))

        self.wait(1)

        tex_plus = MathTex("+").scale(1).next_to(tex1, DOWN).shift(0.2 * UP)
        self.play(FadeIn(tex_plus))

        vg_plus = VGroup(tex1, tex2, tex_plus)

        tex_eq = MathTex(" = 1").scale(1).next_to(vg_plus, RIGHT)
        vg_plus.add(tex_eq)
        self.play(Create(tex_eq))
        self.wait(1)

        tex_final = MathTex("P(x\geq2)", "=", "1", "-", "P(x=1)").next_to(vg_svg_copy, DOWN)
        self.play(TransformMatchingTex(vg_plus, tex_final))
        self.wait(1)
        self.tex = tex_final

        self.play(FadeOut(vg_svg_copy),
                  FadeOut(tex_final))
        # self.play(tex_final.animate.to_edge(RIGHT))
        self.wait(1)


class ProbCompute(Scene):
    def construct(self):
        self.intro_problem()
        self.intro_calendar()
        self.intro_prob()
        self.intro_math()

    def intro_problem(self):
        n_svg = 24
        vg_svg = VGroup(*[SVGMobject('svg_icon/people.svg', fill_color=WHITE).scale(0.5) for _ in range(n_svg)])

        vg_svg.arrange_in_grid(rows=4, buff=0.6).to_edge(LEFT)
        self.vg_svg = vg_svg

    def intro_calendar(self):
        n_circles = 365
        vg_circle = VGroup(*[Circle(radius=0.3, color=BLUE).scale(0.6) for _ in range(n_circles)])
        vg_circle.arrange_in_grid(cols=12, buff=0.1).scale(0.45).to_edge(RIGHT)

        self.vg_calendar = vg_circle

    def intro_prob(self):
        vg_prob = VGroup(*[MathTex("\\frac{365-%d}{365}" % i, color=YELLOW).scale(0.45) for i in range(24)])
        self.play(FadeIn(self.vg_calendar))
        l = random.sample(range(365), 24)
        for n in range(24):
            vg_prob[n].next_to(self.vg_svg[n], 0.5 * UP)
            self.play(FadeIn(self.vg_svg[n]))
            self.play(self.vg_calendar[l[n]].animate.set_color(YELLOW))
            self.play(Transform(self.vg_calendar[l[n]].copy(), vg_prob[n]))

        self.vg_prob = vg_prob

    def intro_math(self):
        self.play(FadeOut(self.vg_calendar))
        tex_prob = MathTex("\\frac{365-i}{365}").next_to(self.vg_svg, 5 * RIGHT)
        tex_prod = MathTex("\prod_0^{23} ").next_to(tex_prob, LEFT)
        tex_res = MathTex("\\approx 0.462").next_to(tex_prob, RIGHT)

        tex_final = MathTex("P(x\geq 2)", "=", "1", "-", "P(x=1)").scale(0.8).next_to(
            VGroup(tex_prob, tex_prod, tex_res), 1.5 * UP)

        self.play(FadeIn(tex_final))
        self.wait(1)

        self.play(Transform(tex_final[-1].copy(), tex_prob))
        self.wait(1)
        self.play(Indicate(self.vg_prob),
                  Indicate(tex_prob))
        self.wait(1)
        self.play(FadeIn(tex_prod))

        self.play(Write(tex_res))
        self.wait(1)

        self.play(FadeTransform(tex_res.copy(), tex_final[-1]))
        self.wait(1)

        tex_2 = MathTex('\\approx 1-0.462').next_to(VGroup(tex_prob, tex_prod, tex_res), 1.5 * DOWN)
        tex_3 = MathTex("0.538(53.8\%)").next_to(tex_2, DOWN)
        self.play(FadeTransform(tex_final[0].copy(), tex_2))
        self.play(FadeIn(tex_3))
        self.wait(1)
        # tex_final = MathTex("P(x\geq2)", "=", "1", "-", "P(x=1)").next_to(vg_svg_copy, DOWN)


class BirthdayProof(Scene):
    def construct(self):
        self.formula= None
        self.var = None

        #self.prob_formula()
        self.var_prob()
        self.data_bar()

    @staticmethod
    def birth_func(k, n):
        up = k * (k - 1)
        down = 2 * n
        return 1 - np.exp(-up / down)

    @staticmethod
    def birth_prob(k):
        k = int(k)
        s = 1
        for i in range(k):
            s = s * (365 - i) / 365
        return 1-s

    def prob_formula(self):
        tex_example = MathTex("P", "=", "1", "-", "\prod_0^{23}","\\frac{365-i}{365}")
        tex_general = MathTex("\prod_0^{k-1}", color=YELLOW).move_to(tex_example[-1].get_center())

        self.add(tex_example)
        self.play(Transform(tex_example[-2], tex_general))
        self.wait(1)

    def var_prob(self):
        x_var = Variable(1, 'k', var_type=Integer)
        sqr_var = Variable(0, 'P(k)', num_decimal_places=3)

        vg_var = VGroup(x_var, sqr_var)
        vg_var.arrange(DOWN).scale(0.8).to_edge(UP)

        sqr_var.add_updater(lambda v: v.tracker.set_value(self.birth_prob(x_var.tracker.get_value())))
        self.add(vg_var)
        self.var = vg_var

    def data_bar(self):
        values = [self.birth_prob(i) for i in range(1, 60)]
        chart = BarChart(
            values,
            bar_names=[str(i) for i in range(1, 60)],
            y_range=[0, 1, 0.5],
            y_length=6,
            x_length=10,
            x_axis_config={"font_size": 15})
        # ax = CommonFunc.add_axes(x_range=[0, 1], y_range=[1, 50], x_length=10, y_length=6,
        #                          axis_config={"include_tip": False, "include_numbers": False})
        self.add(chart[2])
        self.play(Create(chart[0], run_time=10, rate_func=linear),
                  self.var[0].tracker.animate.set_value(60), run_time=10, rate_func=linear)
        self.wait(2)

    def data_prepare(self):
        ax = CommonFunc.add_axes(x_range=[0, 101], y_range=[0, 1.0], x_length=8, y_length=6,
                                 axis_config={"include_tip": False, "include_numbers": False}).scale(1)
        self.play(Create(ax))
        fit_plot = ax.plot(lambda x: self.birth_func(k=x, n=365), x_range=[1, 100], use_smoothing=True, color=YELLOW)

        self.play(Create(fit_plot))
        self.wait(2)
