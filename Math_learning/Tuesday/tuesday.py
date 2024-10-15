# -*- coding: utf-8 -*-
from datetime import datetime
import random
from manim import *
import numpy as np
import sys

random.seed(17)
sys.path.append('..')

from CS_learning.common_func import CommonFunc


class TuesdayParadox(Scene):
    def construct(self):
        self.vg_problem = VGroup()

        self.intro_problem()
        self.problem_anly()

    def intro_problem(self):
        n_circles = 2
        circles = VGroup(*[Circle(radius=0.8,
                                  color=YELLOW)for _ in range(n_circles)])
        circles.arrange_submobjects(RIGHT, buff=1).to_edge(3*DOWN)

        svg_people = SVGMobject('svg_icon/parents.svg', fill_color=WHITE).to_edge(UP)

        arrow1 = Arrow(start=svg_people.get_edge_center(DOWN), end=circles[0].get_edge_center(UP), color=WHITE)
        arrow2 = Arrow(start=svg_people.get_edge_center(DOWN), end=circles[1].get_edge_center(UP), color=WHITE)

        self.play(FadeIn(svg_people))
        self.play(GrowFromPoint(arrow1, svg_people),
                  GrowFromPoint(arrow2, svg_people))

        self.play(FadeIn(circles))
        self.wait(1)

        #svg_girl = SVGMobject('svg_icon/girl.svg', fill_color=WHITE)
        svg_boy = SVGMobject('svg_icon/boy.svg', fill_color=BLUE).scale(0.55).move_to(circles[0].get_center())

        self.play(FadeIn(svg_boy))
        self.wait(1)

        svg_2 = SVGMobject('svg_icon/boy.svg', fill_color=GRAY).scale(0.55).move_to(circles[1].get_center())
        self.play(SpinInFromNothing(svg_2))
        self.wait(1)

        vg_boy = VGroup(svg_boy, svg_2)

        self.vg_problem.add(svg_people, arrow1, arrow2, circles, vg_boy)

    def problem_anly(self):
        self.play(self.vg_problem.animate.to_edge(LEFT))

        answer1 = MathTex("P=\\frac{1}{2}")
        answer2 = MathTex("P=\\frac{1}{3}")

        svg_boy = SVGMobject('svg_icon/boy.svg', fill_color=BLUE).scale(0.55)
        svg_girl = SVGMobject('svg_icon/girl.svg', fill_color=RED).scale(0.55)
        t0 = MobjectTable([[svg_boy.copy(), svg_girl.copy()]])

        t1 = MobjectTable([[VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(RIGHT), VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(LEFT)],
                           [VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT), VGroup(svg_girl.copy(), svg_girl.copy()).arrange_submobjects(RIGHT)]])

        vg_answer = VGroup(answer1, t0, answer2, t1).arrange_in_grid(2,2).to_edge(RIGHT)

        self.play(FadeIn(answer1, target_position=self.vg_problem[-2][-1]),
                  FadeIn(answer2, target_position=self.vg_problem[-2][-1]))
        self.wait(1)
        self.play(FadeIn(t0, target_position=answer1))
        self.wait(1)
        self.play(FadeIn(t1, target_position=answer2))
        self.wait(1)

        #cell = t1.get_cell((2, 2))
        cell = t1.get_entries((2, 2))
        self.play(FadeOut(cell))

        self.wait(2)

        #self.play(FadeIn(vg_answer))

class TuesdayAnaly(Scene):
    def construct(self):
        self.vg_table = None

        self.get_table()

    def get_table(self):
        l_week = ['周一 男', '周二 男', '...', '周日 男', '周一 女', '...', '周日 女']
        svg_boy = SVGMobject('svg_icon/boy.svg', fill_color=BLUE).scale(0.3)
        svg_girl = SVGMobject('svg_icon/girl.svg', fill_color=RED).scale(0.3)
        a_sample = [[VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                    VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                    VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                    VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                    VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(RIGHT),
                    VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(RIGHT),
                    VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(RIGHT)],

                    [VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                    VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                    VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                    VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(RIGHT)],

                    [VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                    VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                    VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                    VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(RIGHT)],

                    [VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(RIGHT)],

                    [VGroup(svg_girl.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_girl.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_girl.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_girl.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_girl.copy(), svg_girl.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_girl.copy(), svg_girl.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_girl.copy(), svg_girl.copy()).arrange_submobjects(RIGHT)],

                    [VGroup(svg_girl.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_girl.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_girl.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_girl.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_girl.copy(), svg_girl.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_girl.copy(), svg_girl.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_girl.copy(), svg_girl.copy()).arrange_submobjects(RIGHT)],

                    [VGroup(svg_girl.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_girl.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_girl.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_girl.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_girl.copy(), svg_girl.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_girl.copy(), svg_girl.copy()).arrange_submobjects(RIGHT),
                     VGroup(svg_girl.copy(), svg_girl.copy()).arrange_submobjects(RIGHT)]
                    ]
        #t0 = MobjectTable(row_labels=labels, col_labels=labels, table=[[Text('10').scale(0.1)]*14]*14)
        t0 = MobjectTable(a_sample,
                          col_labels=[Text(i) for i in l_week],
                          row_labels=[Text(i) for i in l_week]).scale(0.5)

        self.play(FadeIn(t0))
        self.wait(2)

        self.vg_table = t0
    def table_analy(self):
        pass



