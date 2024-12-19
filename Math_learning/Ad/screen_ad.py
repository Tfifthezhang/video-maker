# -*- coding: utf-8 -*-
from datetime import datetime
import random
from manim import *
import numpy as np
import sys

np.random.seed(17)
sys.path.append('..')

from CS_learning.common_func import CommonFunc


class ScreenFunc(Scene):
    def construct(self):
        self.rec = None
        self.tex = None

        self.intro_screen()
        self.proof1()
        #self.curve()

    @staticmethod
    def area_func(x, c=27):
        return x/(x**2+1)*c**2

    @staticmethod
    def get_rectangle_corners(bottom_left, top_right):
        return [
            (top_right[0], top_right[1],0),
            (bottom_left[0], top_right[1],0),
            (bottom_left[0], bottom_left[1],0),
            (top_right[0], bottom_left[1],0),
        ]

    def new_rec(self, dia):
        start_point = dia.get_start()
        end_point = dia.get_end()
        res_coor = self.get_rectangle_corners(start_point, end_point)
        polygon = Polygon(*res_coor, color=WHITE)
        return polygon

    def intro_screen(self):

        rect = Rectangle(width=6.0, height=4.0, color=WHITE)

        dia = Line(start=rect.get_corner(DL), end=rect.get_corner(UR), color=MAROON)

        poly_rec = self.new_rec(dia)

        self.play(Create(poly_rec))

        self.wait(2)

        self.rec = VGroup(dia,poly_rec)

        length1 = MathTex('a').scale(0.8).next_to(poly_rec, DOWN)
        length2 = MathTex('b').scale(0.8).next_to(poly_rec, LEFT)

        length1.add_updater(lambda x: x.next_to(poly_rec, DOWN))
        length2.add_updater(lambda x: x.next_to(poly_rec, LEFT))

        self.play(Write(length1),
                  Write(length2))

        len_dia = MathTex('c', '=', '\sqrt{a^2+b^2}', color=MAROON).scale(0.9)
        s_area = MathTex('S', '=', 'ab', color=BLUE).scale(0.9)

        vg_tex = VGroup(len_dia, s_area).arrange_submobjects(DOWN, buff=1).to_edge(RIGHT)

        self.play(SpinInFromNothing(dia))
        self.play(FadeTransform(dia.copy(), vg_tex[0]))

        self.play(poly_rec.animate.set_fill(BLUE, 1))
        self.play(FadeTransform(poly_rec.copy(), vg_tex[1]))
        self.wait(1)

        self.tex = VGroup(length1, length2, len_dia, s_area)

        #t = ValueTracker(0)

        poly_rec.add_updater(lambda x: x.become(self.new_rec(dia)))
        self.play(Rotate(dia, angle=PI, about_point=ORIGIN), run_time=10)
        # dia.add_updater(lambda x, dt: x.become(x.copy()).rotate(t.get_value()*DEGREES*dt))
        poly_rec.add_updater(lambda x: x.become(self.new_rec(dia)))

    def proof1(self):
        r =3.6742
        circ = Circle(radius=r, color=PURPLE)
        self.play(FadeIn(circ))

        self.wait(2)


    def curve(self):
        ax = CommonFunc.add_axes(x_range=[0.5, 10, 0.5], y_range=[0.2, 500, 100], x_length=10, y_length=6,
                                 axis_config={"include_tip": False, "include_numbers":True}).scale(1)
        self.play(Create(ax))
        fit_plot = ax.plot(lambda x: self.area_func(x), x_range=[0.5, 10], use_smoothing=True, color=YELLOW)

        self.play(Create(fit_plot))

        fit_plot2 = ax.plot(lambda x: self.area_func(x, c=28), x_range=[0.5, 10], use_smoothing=True, color=GREEN)

        self.play(Create(fit_plot2))

        fit_plot3 = ax.plot(lambda x: self.area_func(x, c=26), x_range=[0.5, 10], use_smoothing=True, color=MAROON)

        self.play(Create(fit_plot3))

        self.wait(2)

        lines_1 = ax.get_lines_to_point(circ.get_right(), color=GREEN_B)
