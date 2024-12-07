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
        self.sex = None
        self.week = None

        self.display()
        #self.processing()
    def display(self):
        svg_boy = SVGMobject('svg_icon/boy.svg', fill_color=BLUE).scale(0.55)
        svg_girl = SVGMobject('svg_icon/girl.svg', fill_color=RED).scale(0.55)

        l_week = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        circles = VGroup(*[Circle(color=YELLOW).scale(0.55) for _ in range(7)]).arrange_submobjects(RIGHT,buff=0.3)
        vg_text = VGroup(*[Text(l_week[i]).scale(0.6).move_to(circles[i].get_center()) for i in range(7)])
        vg_week1 = VGroup(circles, vg_text).scale(0.6)
        vg_week2 = vg_week1.copy()

        VGroup(svg_boy, vg_week1, svg_girl,vg_week2).arrange_in_grid(rows=2, buff=1).to_edge(2*UP)

        self.play(Create(svg_boy),
                  Create(svg_girl),
                  Create(vg_week1),
                  Create(vg_week2))

        self.wait(2)

        # self.sex = vg_sex
        # self.week = vg_week

    def processing(self):

        vg_sex = self.sex
        vg_week = self.week[0]

        a_week = np.random.randint(low=0, high=7, size=(100,2))
        a_sex = np.random.randint(0, 2, size=(100, 2))

        tracker1 = ValueTracker(0)
        pointer = Vector(UP).next_to(vg_sex[int(tracker1.get_value())], DOWN)
        pointer.add_updater(lambda m: m.next_to(vg_sex[int(tracker1.get_value())], DOWN))

        self.play(FadeIn(pointer))

        self.play(tracker1.animate.set_value(1))

        self.wait(1)

        tracker2 = ValueTracker(0)
        pointer_week = Vector(UP).next_to(vg_week[int(tracker2.get_value())], DOWN)
        pointer_week.add_updater(lambda m: m.next_to(vg_week[int(tracker2.get_value())], DOWN))

        self.play(FadeIn(pointer_week))

        self.play(tracker2.animate.set_value(6))

        self.wait(1)

        m = Variable(0, label=MathTex('m'), var_type=Integer)

        n = Variable(0, label=MathTex('n'), var_type=Integer)

        VGroup(n, m).arrange_submobjects(DOWN,buff=1).to_edge(2*DOWN)

        self.play(Write(m))
        self.play(Write(n))
        for _ in range(100):
            i_sex = np.random.randint(2)
            self.play(tracker1.animate.set_value(i_sex))
            i_week = np.random.randint(7)
            self.play(tracker2.animate.set_value(i_week))

            self.play(m.tracker.animate.set_value(m.tracker.get_value() + 1))
            if i_sex==0 and i_week==1:
                self.play(n.tracker.animate.set_value(n.tracker.get_value() + 1))
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


