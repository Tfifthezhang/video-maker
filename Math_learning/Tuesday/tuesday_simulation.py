# -*- coding: utf-8 -*-
from datetime import datetime
import random
from manim import *
import numpy as np
import sys

np.random.seed(17)
sys.path.append('..')

from CS_learning.common_func import CommonFunc


class ThusdaySim(Scene):
    def construct(self):
        self.vg_sample = None
        self.vg_chart = None
        self.pointer = None
        self.c_label = None

        self.display()
        self.table_dis()
        self.processing()
    def display(self):
        svg_boy = SVGMobject('svg_icon/boy.svg', fill_color=BLUE).scale(0.55)
        svg_girl = SVGMobject('svg_icon/girl.svg', fill_color=RED).scale(0.55)
        vg_sex = VGroup(svg_boy, svg_girl).arrange_submobjects(DOWN, buff=1)

        l_week = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        circles = VGroup(*[Circle(color=YELLOW).scale(0.55) for _ in range(7)]).arrange_submobjects(RIGHT,buff=0.3)
        vg_text = VGroup(*[Text(l_week[i]).scale(0.6).move_to(circles[i].get_center()) for i in range(7)])
        vg_week = VGroup(circles, vg_text).scale(0.7)

        vg_sample = VGroup(vg_sex,vg_week).arrange_submobjects(RIGHT).to_edge(0.5*UP)

        self.play(Create(vg_sample))

        self.wait(2)

        self.vg_sample = vg_sample

        pointer1 = Vector(RIGHT,color=TEAL_A).next_to(svg_boy, LEFT)
        pointer2 = Vector(UP, color=TEAL_A).next_to(circles[0], DOWN)
        self.play(pointer1.animate.next_to(svg_girl, LEFT),
                  pointer2.animate.next_to(circles[-1], DOWN))
        self.wait(1)

        self.pointer = VGroup(pointer1, pointer2)

    def table_dis(self):

        values_sex = [0]*4
        chart1 = BarChart(
            values_sex,
            bar_names=['bb', 'bg', 'gb', 'gg'],
            y_range=[0, 30, 10],
            y_length=6,
            x_length=10,
            x_axis_config={"font_size": 50})

        values_week = [0] * 2
        chart2 = BarChart(
            values_week,
            bar_names=['One of them is a boy, born on Tuesday',
                       'The remaining one is a boy'],
            y_range=[0, 20, 10],
            y_length=6,
            x_length=10,
            x_axis_config={"font_size": 30})

        vg_chart = VGroup(chart1, chart2).arrange_submobjects(RIGHT, buff=2).scale(0.5).next_to(self.vg_sample, DOWN)
        self.play(Create(vg_chart))
        self.wait(2)

        self.vg_chart = vg_chart

        c_bar_lbls = chart2.get_bar_labels(font_size=48)
        self.play(Write(c_bar_lbls))
        self.wait(1)

        self.c_label = c_bar_lbls

        # answer = MathTex("\\frac{7}{15}").scale(1)
        # cal_P = MathTex("\\approx", "0.467").scale(0.8).next_to(answer, 1.2 * RIGHT)
        #
        # VGroup(answer, cal_P).arrange_submobjects(RIGHT).to_edge(RIGHT).shift(2*UP)
        # self.play(ReplacementTransform(c_bar_lbls, answer))
        # self.play(Write(cal_P))
        # self.wait(1)

    def processing(self):
        np.random.seed(17)
        vg_sex = self.vg_sample[0]
        circles = self.vg_sample[1][0]
        chart1, chart2 = self.vg_chart
        p1, p2 = self.pointer

        a_week = np.random.randint(low=0, high=7, size=(200, 2))
        a_sex = np.random.randint(0, 2, size=(200, 2))

        d_chart1_value = {'0': 0, '1': 0, '2': 0, '3': 0}
        d_chart2_value = {'O': 0, 'T': 0}

        for i in range(200):
            l_week = a_week[i]
            l_sex = a_sex[i]

            if sum(l_sex) == 0:
                d_chart1_value['0'] += 1
            if sum(l_sex) == 2:
                d_chart1_value['3'] += 1
            if sum(l_sex) == 1:
                if l_sex[0] == 0:
                    d_chart1_value['1'] += 1
                if l_sex[0] == 1:
                    d_chart1_value['2'] += 1

            if (l_week[0] == 1 and l_sex[0] == 0) or (l_week[1] == 1 and l_sex[1] == 0):
                d_chart2_value['O'] += 1
                if sum(l_sex) == 0:
                    d_chart2_value['T'] += 1

            for j in range(2):
                sex = l_sex[j]
                week = l_week[j]
        #         self.play(p1.animate.next_to(vg_sex[sex], LEFT),
        #                   p2.animate.next_to(circles[week], DOWN))
        #
        #     self.play(chart1.animate.change_bar_values(list(d_chart1_value.values())),
        #               chart2.animate.change_bar_values(list(d_chart2_value.values())),
        #               Transform(self.c_label,chart2.get_bar_labels(font_size=48)))
        print(d_chart1_value, d_chart2_value)
        # self.wait(1)

        answer = MathTex("\\frac{7}{15}").scale(1)
        cal_P = MathTex("\\approx", "0.467").scale(0.8).next_to(answer, 1.2 * RIGHT)

        VGroup(answer, cal_P).arrange_submobjects(RIGHT).to_edge(RIGHT).shift(2*UP)
        self.play(ReplacementTransform(self.c_label.copy(), answer))
        self.play(Write(cal_P))
        self.wait(1)

class GeoSim(Scene):
    def construct(self):

        self.ax = None
        self.dots = None

        self.display()

    def display(self):
        ax = NumberPlane(x_range=[0, 1], y_range=[0, 7], x_length=8, y_length=6,
                         background_line_style={
                             "stroke_color": TEAL,
                             "stroke_width": 4,
                             "stroke_opacity": 0.8,
                         },
                         axis_config={"include_tip": False, "include_numbers": False})
        self.play(Create(ax))

        a_y = np.random.uniform(low=0, high=7, size=(100,2))
        a_x = np.random.uniform(low=0, high=1, size=(100,2))
        a_sex = np.random.randint(0, 2, size=(100, 2))

        d_sex_color = {0: BLUE, 1: RED}

        vg_tuesday_boy = VGroup()
        vg_answer = VGroup()
        vg_dots = VGroup()
        for i in range(100):
            x = a_x[i]
            y = a_y[i]
            sex = a_sex[i]
            dot1 = Dot(ax.c2p(x[0], y[0]), radius=1 * DEFAULT_DOT_RADIUS, stroke_width=8, fill_opacity=.5, color = d_sex_color[a_sex[i][0]])
            dot2 = Dot(ax.c2p(x[1], y[1]), radius=1 * DEFAULT_DOT_RADIUS, stroke_width=8, fill_opacity=.5, color = d_sex_color[a_sex[i][1]])
            line = Line(dot1, dot2, color=WHITE, fill_opacity=0.1, stroke_width=0.3)
            vg_dot = VGroup(dot1, dot2, line)
            self.play(FadeIn(vg_dot))

            vg_dots.add(vg_dot)

            if (1 <= y[0] <= 2 and sex[0] == 0) or (1 <= y[1] <= 2 and sex[1] == 0):
                vg_tuesday_boy.add(vg_dot)
                if sum(sex) == 0:
                    vg_answer.add(vg_dot)

        self.wait(1)

        self.play(VGroup(ax, vg_dots).animate.to_edge(LEFT))
        self.wait(1)

        self.play(FadeTransform(vg_dots, vg_tuesday_boy))
        self.wait(1)

        vg_line_tuesday = VGroup(*[i[-1] for i in vg_tuesday_boy])
        vg_line_answer = VGroup(*[i[-1] for i in vg_answer])

        a = vg_line_tuesday.copy().scale(0.4)
        b = vg_line_answer.copy().scale(0.6)
        tuesday_n = MathTex('%d' % len(vg_line_tuesday))
        answer_n = MathTex('%d' % len(vg_line_answer))

        VGroup( b, answer_n, a, tuesday_n,).arrange_in_grid(rows=2, buff=1).to_edge(RIGHT)

        self.play(Indicate(vg_line_tuesday))
        self.play(FadeIn(a, target_position=vg_line_tuesday))
        self.play(FadeIn(tuesday_n, target_position=a))
        self.wait(1)
        self.play(Indicate(vg_line_answer))
        self.play(FadeIn(b, target_position=vg_line_answer))
        self.play(FadeIn(answer_n, target_position=b))

        self.wait(1)

        answer = MathTex("\\frac{7}{15}").scale(0.8).to_edge(5*RIGHT)
        self.play(ReplacementTransform(VGroup(answer_n.copy(), tuesday_n.copy()),answer))
        cal_P = MathTex("\\approx", "0.467").scale(0.8).next_to(answer, 1.2*RIGHT)
        self.play(Write(cal_P))

        self.ax = ax
        self.dots = [a_x, a_y, a_sex]

    def filter(self):
        a_x, a_y, a_sex, dots = self.dots
        ax = self.ax

class GeoGen(ThreeDScene):
    #np.random.seed(42)
    def construct(self):
        self.ax = None
        self.dots = None

        vg_scene = VGroup()
        for i in range(10):
            np.random.seed(i)
            res = self.display().scale(0.5)
            vg_scene.add(res)

        vg_scene.arrange_in_grid(rows=2,buff=1).scale(0.35)
        #self.move_camera(frame_center=vg_scene[0], zoom=1)
        self.add(vg_scene)
        self.wait(1)
        self.move_camera(frame_center=vg_scene[0], zoom=2.85*2)
        self.wait(1)
        for j in range(1, 10):
            self.move_camera(frame_center=vg_scene[j], zoom=2.85*2)
            self.wait(1)
        self.move_camera(frame_center=ORIGIN, zoom=1)
        self.wait(1)

        vg_prob = VGroup(*[i[-2] for i in vg_scene])
        self.play(vg_scene.animate.shift(1.5*UP))
        self.play(vg_prob.animate.scale(3.5).arrange_submobjects(RIGHT, buff=1).next_to(vg_scene, DOWN))
        self.wait(1)
        math_res = MathTex("\\frac{10}{16}+","\\frac{8}{19}+","\\frac{10}{18}+","\\frac{10}{15}+","\\frac{8}{13}+",
                           "\\frac{5}{16}+","\\frac{7}{17}+","\\frac{6}{14}+","\\frac{3}{12}+","\\frac{9}{17}").scale(0.55).next_to(vg_scene, 0.1*DOWN)
        brace = Brace(math_res, DOWN, color=MAROON_A)
        math_final = MathTex("\\frac{T}{10}").scale(0.8).next_to(brace, DOWN)
        math_cal = MathTex("\\approx", '0.48159').scale(0.8).next_to(math_final, 1.2*RIGHT)

        self.play(ReplacementTransform(vg_prob, math_res))
        self.play(Create(brace),
                  Write(math_final))
        self.play(Write(math_cal))
        self.wait(1)

        std_res = MathTex("\\frac{13}{27}", "\\approx", "0.48148").scale(0.8).next_to(brace, LEFT)
        self.play(FadeIn(std_res))
        self.play(math_cal[-1].animate.set_color(GREEN),
                  std_res[-1].animate.set_color(GREEN))
        self.wait(1)
    def display(self):
        ax = NumberPlane(x_range=[0, 1], y_range=[0, 7], x_length=8, y_length=6,
                         background_line_style={
                             "stroke_color": TEAL,
                             "stroke_width": 4,
                             "stroke_opacity": 0.8,
                         },
                         axis_config={"include_tip": False, "include_numbers": False}).to_edge(LEFT)
        a_y = np.random.uniform(low=0, high=7, size=(100, 2))
        a_x = np.random.uniform(low=0, high=1, size=(100, 2))
        a_sex = np.random.randint(0, 2, size=(100, 2))

        d_sex_color = {0: BLUE, 1: RED}

        vg_tuesday_boy = VGroup()
        vg_answer = VGroup()
        vg_dots = VGroup()
        for i in range(100):
            x = a_x[i]
            y = a_y[i]
            sex = a_sex[i]
            dot1 = Dot(ax.c2p(x[0], y[0]), radius=1 * DEFAULT_DOT_RADIUS, stroke_width=8, fill_opacity=.5, color = d_sex_color[a_sex[i][0]])
            dot2 = Dot(ax.c2p(x[1], y[1]), radius=1 * DEFAULT_DOT_RADIUS, stroke_width=8, fill_opacity=.5, color = d_sex_color[a_sex[i][1]])
            line = Line(dot1, dot2, color=WHITE, fill_opacity=0.1, stroke_width=0.3)
            vg_dot = VGroup(dot1, dot2, line)
            vg_dots.add(vg_dot)
            if (1 <= y[0] <= 2 and sex[0] == 0) or (1 <= y[1] <= 2 and sex[1] == 0):
                vg_tuesday_boy.add(vg_dot)
                if sum(sex) == 0:
                    vg_answer.add(vg_dot)

        vg_line_tuesday = VGroup(*[i[-1] for i in vg_tuesday_boy])
        vg_line_answer = VGroup(*[i[-1] for i in vg_answer])

        a = vg_line_tuesday.copy().scale(0.4)
        b = vg_line_answer.copy().scale(0.4)
        tuesday_n = MathTex('%d' % len(vg_line_tuesday))
        answer_n = MathTex('%d' % len(vg_line_answer))

        VGroup( b, answer_n, a, tuesday_n,).arrange_in_grid(rows=2, buff=1).to_edge(RIGHT)

        answer = MathTex("\\frac{%d}{%d}"%(len(vg_line_answer),len(vg_line_tuesday))).scale(0.8).to_edge(5*RIGHT)
        s= len(vg_line_answer) / len(vg_line_tuesday)
        print(s)
        cal_P = MathTex("\\approx", "%.3f" %s).scale(0.8).next_to(answer, 1.2*RIGHT)

        return VGroup(ax, vg_dots, b, answer_n, a, tuesday_n, answer, cal_P)


