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
        self.vg_answer = None
        self.vg_another = VGroup()

        self.intro_problem()
        self.problem_anly()
        self.another_problem()
        self.problem_trans()

    def intro_problem(self):
        n_circles = 2
        circles = VGroup(*[Circle(radius=0.8,
                                  color=YELLOW)for _ in range(n_circles)])
        circles.arrange_submobjects(RIGHT, buff=0.5).to_edge(3*DOWN)
        rectangle = SurroundingRectangle(circles, color=YELLOW)

        svg_people = SVGMobject('svg_icon/parents.svg', fill_color=WHITE).to_edge(UP)
        arrow = Arrow(start=svg_people.get_edge_center(DOWN), end=rectangle.get_edge_center(UP), color=WHITE)

        self.play(FadeIn(svg_people))
        self.play(GrowFromPoint(arrow, svg_people))

        self.play(FadeIn(rectangle))

        self.wait(1)

        #svg_girl = SVGMobject('svg_icon/girl.svg', fill_color=WHITE)
        svg_boy = SVGMobject('svg_icon/boy.svg', fill_color=BLUE).scale(0.55).move_to(circles[0].get_center())

        self.play(FadeIn(svg_boy))
        self.wait(1)

        svg_2 = SVGMobject('svg_icon/boy.svg', fill_color=GRAY).scale(0.55).move_to(circles[1].get_center())
        self.play(SpinInFromNothing(svg_2))
        self.wait(1)

        vg_boy = VGroup(svg_boy, svg_2)

        self.vg_problem.add(svg_people, arrow, rectangle, vg_boy)

    def problem_anly(self):
        self.play(self.vg_problem.animate.to_edge(LEFT))

        answer1 = MathTex("P=\\frac{1}{2}")
        answer2 = MathTex("P=\\frac{1}{3}")

        svg_boy = SVGMobject('svg_icon/boy.svg', fill_color=BLUE).scale(0.55)
        svg_girl = SVGMobject('svg_icon/girl.svg', fill_color=RED).scale(0.55)
        t0 = MobjectTable([[svg_boy.copy(), svg_girl.copy()]])

        t1 = MobjectTable([[VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(RIGHT), VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(LEFT)],
                           [VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT), VGroup(svg_girl.copy(), svg_girl.copy()).arrange_submobjects(RIGHT)]])

        vg_answer = VGroup(answer1, t0, answer2, t1).arrange_in_grid(2, 2).to_edge(RIGHT)

        self.play(FadeIn(answer1, target_position=self.vg_problem[-2][-1]),
                  FadeIn(answer2, target_position=self.vg_problem[-2][-1]))
        self.wait(1)
        self.play(FadeIn(t0, target_position=answer1))
        self.wait(1)
        self.play(FadeIn(t1, target_position=answer2))
        self.wait(1)

        cell = t1.get_entries((2, 2))
        self.play(FadeOut(cell))

        self.wait(2)

        self.vg_answer = vg_answer

    def another_problem(self):
        n_circles = 2
        circles = VGroup(*[Circle(radius=0.8,
                                  color=YELLOW)for _ in range(n_circles)])
        circles.arrange_submobjects(RIGHT, buff=0.5).to_edge(3*DOWN)

        svg_people = SVGMobject('svg_icon/parents.svg', fill_color=WHITE).to_edge(UP)

        arrow1 = Arrow(start=svg_people.get_edge_center(DOWN), end=circles[0].get_edge_center(UP), color=WHITE)
        arrow2 = Arrow(start=svg_people.get_edge_center(DOWN), end=circles[1].get_edge_center(UP), color=WHITE)

        #svg_girl = SVGMobject('svg_icon/girl.svg', fill_color=WHITE)
        svg_boy = SVGMobject('svg_icon/boy.svg', fill_color=BLUE).scale(0.55).move_to(circles[0].get_center())

        svg_2 = SVGMobject('svg_icon/boy.svg', fill_color=GRAY).scale(0.55).move_to(circles[1].get_center())

        vg_boy = VGroup(svg_boy, svg_2)

        vg_another = VGroup()
        vg_another.add(svg_people, arrow1, arrow2, circles, vg_boy)
        vg_another.to_edge(LEFT)

        self.play(Transform(self.vg_problem[1], vg_another[1:3]),
                  Transform(self.vg_problem[2], vg_another[3]),
                  Transform(self.vg_problem[-1], vg_another[-1]))

    def problem_trans(self):
        answer1, t0, answer2, t1 = self.vg_answer

        cell = t1.get_entries((1, 2))
        self.play(FadeOut(cell))
        answer3 = MathTex("P=\\frac{1}{2}").move_to(answer2)
        self.play(Transform(answer2, answer3))

        self.wait(2)

class TuesdayProblem(Scene):
    def construct(self):
        self.prob_contra()
    def prob_contra(self):
        text1 = Text("其中一个是男孩").scale(0.7)
        en_text1 = Text("One of them is a boy").scale(0.4).next_to(text1, DOWN)

        vg_text1 = VGroup(text1, en_text1).to_edge(2*UP+LEFT)

        text2 = Text('第一个是男孩').scale(0.7).next_to(vg_text1,DOWN)
        en_text2 = Text("The first one is a boy").scale(0.4).next_to(text2, DOWN)

        vg_text2 = VGroup(text2, en_text2)

        self.play(FadeIn(vg_text1), FadeIn(vg_text2))

        ellipse1 = Ellipse(width=5.0, height=6.0, fill_opacity=0.8, color=BLUE)
        ellipse2 = Ellipse(width=2.0, height=3.0, fill_opacity=1, color=RED)

        self.play(FadeTransform(vg_text1.copy(), ellipse1))
        self.play(FadeTransform(vg_text2.copy(), ellipse2))

        self.wait(2)

class TuesdayAnaly(Scene):
    def construct(self):
        self.vg_table = None
        self.vg_problem = VGroup()

        #self.intro_problem()
        #self.problem_anly()
        self.get_table()
        self.table_analy()

    def intro_problem(self):
        n_circles = 2
        circles = VGroup(*[Circle(radius=0.8,
                                  color=YELLOW)for _ in range(n_circles)])
        circles.arrange_submobjects(RIGHT, buff=0.5).to_edge(3.8*DOWN)
        rectangle = SurroundingRectangle(circles, color=YELLOW)

        svg_people = SVGMobject('svg_icon/parents.svg', fill_color=WHITE).to_edge(UP)
        arrow = Arrow(start=svg_people.get_edge_center(DOWN), end=rectangle.get_edge_center(UP), color=WHITE)

        self.play(FadeIn(svg_people))
        self.play(GrowFromPoint(arrow, svg_people))

        self.play(FadeIn(rectangle))

        self.wait(1)
        svg_boy = SVGMobject('svg_icon/boy.svg', fill_color=BLUE).scale(0.55).move_to(circles[0].get_center())
        self.play(FadeIn(svg_boy))
        self.wait(1)

        table_week = Table([["周一", "周二", "周三", "周四", "周五", "周六", "周日"]]).scale(0.4).next_to(rectangle, DOWN)
        table_week.add_highlighted_cell((1, 2), color=MAROON)
        self.play(FadeIn(table_week))

        svg_2 = SVGMobject('svg_icon/boy.svg', fill_color=GRAY).scale(0.55).move_to(circles[1].get_center())
        self.play(SpinInFromNothing(svg_2))
        self.wait(1)

        vg_boy = VGroup(svg_boy, svg_2)

        self.vg_problem.add(svg_people, arrow, rectangle, vg_boy, table_week)

        self.play(self.vg_problem.animate.to_edge(LEFT))

    def problem_anly(self):
        answer1 = MathTex("P\\neq\\frac{1}{2}")
        answer2 = MathTex("P\\neq\\frac{1}{3}")

        svg_boy = SVGMobject('svg_icon/boy.svg', fill_color=BLUE).scale(0.55)
        svg_girl = SVGMobject('svg_icon/girl.svg', fill_color=RED).scale(0.55)
        t0 = MobjectTable([[svg_boy.copy(), svg_girl.copy()]])

        t1 = MobjectTable([[VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(RIGHT), VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(LEFT)],
                           [VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT), VGroup(svg_girl.copy(), svg_girl.copy()).arrange_submobjects(RIGHT)]])

        vg_answer = VGroup(answer1, t0, answer2, t1).arrange_in_grid(2, 2).scale(0.7).to_edge(RIGHT)
        self.play(FadeIn(vg_answer))
        self.wait(1)

        self.play(vg_answer.animate.to_edge(UP))

        # 这个信息没有被使用
        table_week = self.vg_problem[-1]
        self.play(table_week.copy().animate.next_to(vg_answer, DOWN))
        self.wait(1)

        # 综合考虑可能性
        tex_count = MathTex("7", "\\times", "2", "\\times", "2").scale(1.5).next_to(vg_answer, 5*DOWN)
        self.play(FadeIn(tex_count))
        self.play(Indicate(tex_count[0]),
                  Indicate(table_week))

        self.play(Indicate(tex_count[2]),
                  Indicate(t0))

        self.play(Indicate(tex_count[-1]),
                  Indicate(t1.get_entries((1, 2))),
                  Indicate(t1.get_entries((2, 1))))
        self.wait(1)
    def get_table(self):
        self.clear()
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
        t0 = self.vg_table
        target_col = t0.get_columns()[2]
        self.play(Indicate(target_col))
        self.wait(1)
        target_row = t0.get_rows()[2]
        self.play(Indicate(target_row))
        self.wait(1)

        #self.play(target_col.animate.set_color(YELLOW))

        #self.play(t0.add_highlighted_cell((3, 3), color=GREEN))
        #t0.add_highlighted_cell((3, 3), color=GREEN)

        # for i in range(2, 9):
        #     highlight = t0.get_highlighted_cell((i, 3), color=GREEN)
        #     t0.add_to_back(highlight)
        #     #t0.add_highlighted_cell((3, i), color=GREEN)

        self.wait(2)


