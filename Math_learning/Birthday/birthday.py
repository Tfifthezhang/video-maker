# -*- coding: utf-8 -*-
from datetime import datetime
import random
from manim import *
import numpy as np
import sys

random.seed(17)
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
        self.intro_intuition()
        self.intro_combine()
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
        vg_circle.arrange_in_grid(cols=12, buff=0.1).scale(0.3).to_edge(RIGHT)

        brace_in = Brace(vg_circle, direction=LEFT, color=MAROON)
        text_day = Tex('365').scale(0.7).next_to(brace_in, 0.3 * LEFT)

        self.play(Create(vg_circle))
        self.wait(1)

        self.play(Create(brace_in),
                  Write(text_day))
        self.wait(2)

        self.vg_calendar = VGroup(vg_circle, brace_in, text_day)

        brace_out = Brace(self.vg_svg, direction=RIGHT, color=YELLOW)
        text_people = Tex('24').scale(0.6).next_to(brace_out, 0.3 * RIGHT)
        self.play(Create(brace_out),
                  Write(text_people))
        self.wait(1)
        self.vg_brace = VGroup(brace_out, text_people)

    def intro_intuition(self):
        tex_P = MathTex("\\frac{24}{365}").scale(0.6).next_to(self.vg_calendar, UP).shift(2*LEFT)
        self.play(FadeTransform(VGroup(self.vg_brace[-1].copy(),
                                       self.vg_calendar[-1].copy()), tex_P))
        self.wait(1)

        cal_P = MathTex("\\approx", "0.066").scale(0.7).next_to(tex_P, 1.2*RIGHT)
        self.play(Write(cal_P))
        self.wait(1)

        l_tex = [MathTex("\\frac{24-1}{365}").scale(0.6).next_to(self.vg_calendar, UP).shift(2*LEFT),
                 MathTex("\\frac{1-1}{365}").scale(0.6).next_to(self.vg_calendar, UP).shift(2*LEFT),
                 MathTex("\\frac{366-1}{365}").scale(0.6).next_to(self.vg_calendar, UP).shift(2*LEFT)]

        l_cal = [MathTex("0.063").scale(0.7).move_to(cal_P[-1]),
                 MathTex("0").scale(0.7).move_to(cal_P[-1]),
                 MathTex("1").scale(0.7).move_to(cal_P[-1])]

        for i in range(3):
            self.play(Transform(tex_P, l_tex[i]),
                      Transform(cal_P[-1], l_cal[i]))
            self.wait(1)

        self.play(Circumscribe(self.vg_date[2],  color = RED, run_time=1.5),
                  Circumscribe(self.vg_date[-4], color = RED, run_time=1.5))

        self.play(FadeOut(self.vg_date),
                  FadeOut(self.vg_calendar),
                  FadeOut(self.vg_svg),
                  FadeOut(self.vg_brace),
                  FadeOut(tex_P),
                  FadeOut(cal_P))

    def intro_combine(self):
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
        self.tex = None

        self.prob_formula()
        self.var_prob()
        self.data_bar()

    @staticmethod
    def birth_prob(k):
        k = int(k)
        s = 1
        for i in range(k):
            s = s * (365 - i) / 365
        return 1-s

    def prob_formula(self):
        tex_example = MathTex("P", "=", "1", "-", "\prod_1^{23}","\\frac{365-i}{365}")
        tex_general = MathTex("\prod_1^{k-1}", color=YELLOW).move_to(tex_example[-2].get_center())

        self.add(tex_example)
        self.play(Transform(tex_example[-2], tex_general))
        self.wait(2)

        tex_pk = MathTex("P(k)").move_to(tex_example[0].get_center()+0.25*LEFT)
        self.play(Transform(tex_example[0], tex_pk))
        self.wait(2)

        self.tex = tex_example

    def var_prob(self):
        self.play(self.tex.animate.scale(0.8).to_edge(UP+6*LEFT))

        x_var = Variable(1, 'k', var_type=Integer)
        sqr_var = Variable(0, 'P(k)', num_decimal_places=3)

        vg_var = VGroup(x_var, sqr_var)
        vg_var.arrange(DOWN).scale(0.8).next_to(self.tex, DOWN)

        sqr_var.add_updater(lambda v: v.tracker.set_value(self.birth_prob(x_var.tracker.get_value())))
        self.play(FadeIn(vg_var, target_position=self.tex))
        self.var = vg_var

    def data_bar(self):
        values = [self.birth_prob(i) for i in range(1, 61)]
        chart = BarChart(
            values,
            bar_names=[str(i) for i in range(1, 61)],
            y_range=[0, 1, 0.5],
            y_length=6,
            x_length=10,
            x_axis_config={"font_size": 15})
        self.add(chart[2])
        self.play(Create(chart[0], run_time=10, rate_func=linear),
                  Create(chart[1][-1], run_time=10, rate_func=linear),
                  self.var[0].tracker.animate.set_value(60), run_time=10, rate_func=linear)
        self.wait(2)

        ax = CommonFunc.add_axes(x_range=[1, 61], y_range=[0,1,0.5], x_length=10, y_length=6,
                                 axis_config={"include_tip": False, "include_numbers": False})
        self.play(Create(ax))
        self.wait(1)

        line = ax.get_horizontal_line(ax.c2p(61, 0.5), line_func=Line)
        self.play(GrowFromPoint(line, ax.c2p(0, 0.5), RED))
        self.wait(1)

        point = ax.c2p(23.5, self.birth_prob(23)+0.005)
        dot = Dot(point,radius=0.04)
        tex = MathTex('0.507').scale(0.8).next_to(point, UP)

        self.play(Create(dot),
                  Create(tex))

        self.wait(1)

class Math_General(Scene):
    def construct(self):
        self.tex = VGroup()
        self.ax = None

        self.formula_intro()
        self.formula_asymptotic()
        self.introduce_formula()
        self.data_prepare()

        #self.data_prepare()

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

    def formula_intro(self):
        tex = MathTex("P(k)", "=", "1", "-", "\prod_{i=1}^{k-1}", "\\frac{365-i}{365}")
        self.play(Write(tex))
        self.wait(0.5)

        tex_N = MathTex("\\frac{N-i}{N}").move_to(tex[-1].get_center())
        self.play(Transform(tex[-1], tex_N))
        self.wait(0.5)

        tex_pk = MathTex("P(N,k)").move_to(tex[0].get_center()+0.25*LEFT)
        self.play(Transform(tex[0], tex_pk))
        self.wait(1)

        self.play(tex.animate.scale(0.75).to_edge(UP+LEFT))
        self.tex.add(tex)

    def formula_asymptotic(self):
        tex_base = MathTex("1+x","\leq" ,"e^x").to_edge(1.5*UP)
        self.play(Write(tex_base))
        self.tex.add(tex_base)

        ax = CommonFunc.add_axes(x_range=[0, 1], y_range=[0, 3], x_length=10, y_length=6,
                                 axis_config={"include_tip": True, "include_numbers": True}).scale(0.8).to_edge(2*DOWN)

        self.play(Create(ax))
        aux_plot = ax.plot(lambda x: 1+x, x_range=[0, 1], use_smoothing=True, color=RED)
        aux_label = ax.get_graph_label(graph=aux_plot,
                                       label=MathTex('1+x', color=RED)).next_to(aux_plot, 0.1*(UP+RIGHT))
        self.play(Create(aux_plot),
                  Write(aux_label))

        exp_plot = ax.plot(lambda x: np.e**x, x_range=[0, 1], use_smoothing=True, color=YELLOW)
        exp_label = ax.get_graph_label(graph=exp_plot,
                                       label=MathTex('e^x',color=YELLOW)).next_to(exp_plot, 0.1*(UP+RIGHT))

        self.play(Create(exp_plot),
                  Write(exp_label))

        self.ax = VGroup(ax,
                         aux_plot, aux_label,
                         exp_plot, exp_label)

    def introduce_formula(self):
        tex, tex_base = self.tex
        #tex_base = MathTex("1+x", "\leq", "e^x").to_edge(2 * UP)
        tex_N = tex[-1].copy()
        self.play(tex_N.animate.next_to(tex_base[0], DOWN))
        self.wait(1)
        tex_inter = MathTex("1","+","\\frac{-i}{N}").next_to(tex_base[0], DOWN)
        self.play(Transform(tex_N, tex_inter))
        self.wait(1)

        tex_exp = MathTex("\leq","e^\\frac{-i}{N}").next_to(tex_N, 0.2*RIGHT)
        self.play(Write(tex_exp))
        self.wait(1)

        tex_prod = tex[4:].copy()
        self.play(tex_prod.animate.next_to(tex_base, 4*RIGHT))

        tex_leq = MathTex("\leq", "\prod_{i=1}^{k-1} e^\\frac{-i}{N}").next_to(tex_prod, 0.2 * RIGHT)
        self.play(Write(tex_leq))
        self.wait(1)
        self.play(Indicate(tex_leq))
        self.wait(1)
        tex_sum = MathTex("e^\\frac{-k(k-1)}{2N}", color=YELLOW).move_to(tex_leq[-1]).shift(0.2*RIGHT)
        self.play(Transform(tex_leq[-1],tex_sum))
        self.wait(1)

        self.play(FadeOut(tex_N),
                  FadeOut(tex_exp),
                  FadeOut(tex_base))

        self.play(tex.animate.shift(3*RIGHT))

        self.play(Indicate(tex_prod), Indicate(tex[4:]))

        tex_res = MathTex("\geq","1-e^\\frac{-k(k-1)}{2N}", color=YELLOW).scale(0.75).next_to(tex, 0.3*RIGHT)
        self.play(FadeOut(tex_prod),
                  FadeTransform(tex_leq, tex_res))

        self.wait(1)

    def data_prepare(self):
        self.play(FadeOut(self.ax))

        values = [self.birth_prob(i) for i in range(1, 61)]
        chart = BarChart(
            values,
            bar_names=[str(i) for i in range(1, 61)],
            y_range=[0, 1, 0.5],
            y_length=6,
            x_length=10,
            x_axis_config={"font_size": 15})

        self.play(FadeIn(chart))

        ax = CommonFunc.add_axes(x_range=[0, 61], y_range=[0, 1.0], x_length=10, y_length=6,
                                 axis_config={"include_tip": False, "include_numbers": False}).scale(1)
        self.play(Create(ax))
        fit_plot = ax.plot(lambda x: self.birth_func(k=x, n=365), x_range=[1, 61], use_smoothing=True, color=YELLOW)

        self.play(Create(fit_plot))
        self.wait(2)


