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
                                  color=YELLOW) for _ in range(n_circles)])
        circles.arrange_submobjects(RIGHT, buff=0.5).to_edge(3 * DOWN)
        rectangle = SurroundingRectangle(circles, color=YELLOW)

        svg_people = SVGMobject('svg_icon/parents.svg', fill_color=WHITE).to_edge(UP)
        arrow = Arrow(start=svg_people.get_edge_center(DOWN), end=rectangle.get_edge_center(UP), color=WHITE)

        self.play(FadeIn(svg_people))
        self.play(GrowFromPoint(arrow, svg_people))

        self.play(FadeIn(rectangle))

        self.wait(1)

        # svg_girl = SVGMobject('svg_icon/girl.svg', fill_color=WHITE)
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

        t1 = MobjectTable([[VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(RIGHT),
                            VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(LEFT)],
                           [VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                            VGroup(svg_girl.copy(), svg_girl.copy()).arrange_submobjects(RIGHT)]])

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
                                  color=YELLOW) for _ in range(n_circles)])
        circles.arrange_submobjects(RIGHT, buff=0.5).to_edge(3 * DOWN)

        svg_people = SVGMobject('svg_icon/parents.svg', fill_color=WHITE).to_edge(UP)

        arrow1 = Arrow(start=svg_people.get_edge_center(DOWN), end=circles[0].get_edge_center(UP), color=WHITE)
        arrow2 = Arrow(start=svg_people.get_edge_center(DOWN), end=circles[1].get_edge_center(UP), color=WHITE)

        # svg_girl = SVGMobject('svg_icon/girl.svg', fill_color=WHITE)
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

        vg_text1 = VGroup(text1, en_text1).to_edge(2 * UP + LEFT)

        text2 = Text('第一个是男孩').scale(0.7).next_to(vg_text1, DOWN)
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

        # self.intro_problem()
        # self.problem_anly()
        self.get_table()
        self.table_analy()

    def intro_problem(self):
        n_circles = 2
        circles = VGroup(*[Circle(radius=0.8,
                                  color=YELLOW) for _ in range(n_circles)])
        circles.arrange_submobjects(RIGHT, buff=0.5).to_edge(3.8 * DOWN)
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

        t1 = MobjectTable([[VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(RIGHT),
                            VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(LEFT)],
                           [VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                            VGroup(svg_girl.copy(), svg_girl.copy()).arrange_submobjects(RIGHT)]])

        vg_answer = VGroup(answer1, t0, answer2, t1).arrange_in_grid(2, 2).scale(0.7).to_edge(RIGHT)
        self.play(FadeIn(vg_answer))
        self.wait(1)

        self.play(vg_answer.animate.to_edge(UP))

        # 这个信息没有被使用
        table_week = self.vg_problem[-1]
        self.play(table_week.copy().animate.next_to(vg_answer, DOWN))
        self.wait(1)

        # 综合考虑可能性
        tex_count = MathTex("7", "\\times", "2", "\\times", "2").scale(1.5).next_to(vg_answer, 5 * DOWN)
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
        # t0 = MobjectTable(row_labels=labels, col_labels=labels, table=[[Text('10').scale(0.1)]*14]*14)
        t0 = MobjectTable(a_sample,
                          col_labels=[Text(i) for i in l_week],
                          row_labels=[Text(i) for i in l_week]).scale(0.5)

        self.play(t0.create())
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

        self.play(target_col.animate.set_color(YELLOW),
                  target_row.animate.set_color(YELLOW))

        self.wait(2)

        for i in range(2, 6):
            cell_col = t0.get_cell((i, 3))
            cell_row = t0.get_cell((3, i))
            self.play(Flash(cell_col, color=GREEN),
                      Flash(cell_row, color=GREEN))

        self.play(target_col[1:5].animate.set_color(GREEN),
                  (target_row[1:5].animate.set_color(GREEN)))
        self.wait(1)

        self.play(t0.animate.scale(0.95).to_edge(1.2 * UP))
        self.wait(1)

        # 公式推导
        tex_sample = MathTex("28-1", color=YELLOW)
        tex_num = MathTex("14-1", color=GREEN)
        vg_tex = VGroup(tex_sample, tex_num).arrange_submobjects(RIGHT, buff=1.2).next_to(t0, DOWN)
        vg_sample = VGroup(target_col, target_row)
        vg_num = VGroup(target_col[1:5], target_row[1:5])

        self.play(vg_sample.animate.scale(1.4))
        self.wait(1)
        self.play(vg_sample.animate.scale(0.714))
        self.play(Write(vg_tex[0]))
        self.wait(1)

        self.play(vg_num.animate.scale(1.4))
        self.wait(1)
        self.play(vg_num.animate.scale(0.714))
        self.play(Write(vg_tex[1]))
        self.wait(1)

        tex_prob = MathTex("\\frac{14-1}{28-1}").move_to(vg_tex)
        self.play(FadeTransform(vg_tex, tex_prob))
        self.wait(1)

        tex_final = MathTex("=\\frac{13}{27}").next_to(tex_prob, 0.5 * RIGHT)
        self.play(FadeIn(tex_final))
        self.wait(1)


class ProblemGeneral(Scene):
    def construct(self):
        self.table = VGroup()

        self.get_table()
        self.table_display()
        # self.get_func()

    @staticmethod
    def prob_anly(k):
        k = int(k)
        res = (k / 2 - 1) / (k - 1)
        return res

    def get_bar(self):
        values = [self.prob_anly(2 * i) for i in range(1, 100)]
        chart = BarChart(
            values,
            bar_names=[str(i) for i in range(1, 100)],
            y_range=[0, 0.5, 0.5],
            y_length=6,
            x_length=10,
            x_axis_config={"font_size": 15})
        self.play(FadeIn(chart))
        self.wait(2)

    def get_func(self):
        ax = CommonFunc.add_axes(x_range=[0, 100], y_range=[0, 0.5], x_length=10, y_length=6,
                                 axis_config={"include_tip": False, "include_numbers": False}).scale(1)
        self.play(Create(ax))
        fit_plot = ax.plot(lambda x: self.prob_anly(k=x), x_range=[2, 999], use_smoothing=True, color=YELLOW)

        self.play(Create(fit_plot))
        self.wait(2)

    def get_table(self):
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
        b_sample = [[VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
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
        c_sample = [[VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
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
        d_sample = [[VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
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
        l_week = ['周一 男', '周二 男', '...', '周日 男', '周一 女', '...', '周日 女']
        l_month = ['1月 男', '2月 男', '...', '12月 男', '1月 女', '...', '12月 女']
        l_time = ['1点 男', '2点 男', '...', '24点 男', '0点 女', '...', '24点 女']
        l_day = ['1.1 男', '1.2 男', '...', '12.31 男', '1.1 女', '...', '12.31 女']
        samples = [a_sample, b_sample, c_sample, d_sample]
        l_class = [l_week, l_month, l_time, l_day]
        for l in range(4):
            table_ex = MobjectTable(samples[l],
                                    col_labels=[Text(i) for i in l_class[l]],
                                    row_labels=[Text(i) for i in l_class[l]]).scale(0.5)
            target_col = table_ex.get_columns()[2]
            target_row = table_ex.get_rows()[2]
            target_col.set_color(YELLOW)
            target_row.set_color(YELLOW)
            target_col[1:5].set_color(GREEN)
            target_row[1:5].set_color(GREEN)

            self.table.add(table_ex)

        # self.play(l_month.create())
        # self.wait(2)

    def table_display(self):
        vg_table = self.table
        vg_scene = VGroup()
        l_times = [MathTex("7", "\\times", "2", "\\times", "2"),
                   MathTex("12", "\\times", "2", "\\times", "2"),
                   MathTex("24", "\\times", "2", "\\times", "2"),
                   MathTex("365", "\\times", "2", "\\times", "2")]
        l_texts = [Text('男孩在周二出生'), Text('男孩在二月出生'), Text('男孩在两点出生'), Text('男孩在二月二日出生')]
        l_probs = [MathTex("\\frac{14-1}{28-1} = \\frac{13}{27}"),
                   MathTex("\\frac{24-1}{48-1} = \\frac{23}{47}"),
                   MathTex("\\frac{48-1}{96-1} = \\frac{47}{95}"),
                   MathTex("\\frac{730-1}{1460-1} = \\frac{729}{1459}")]
        for i in range(4):
            brace = Brace(vg_table[i], RIGHT)
            l_times[i].next_to(brace, RIGHT)
            l_texts[i].next_to(vg_table[i], UP)
            l_probs[i].scale(0.85).next_to(vg_table[i], DOWN)
            vg_scene.add(VGroup(vg_table[i],brace,l_times[i],l_texts[i],l_probs[i]).scale(0.85).to_edge(1.5*LEFT+0.8*UP))

        #vg_scene.arrange_submobjects(RIGHT, buff=1)
        #vg_scene.arrange_submobjects(RIGHT,buff=1).scale(0.9).shift(vg_scene[0].width*1.5*RIGHT)
        #vg_scene.arrange_in_grid(rows=2, buff=0.5).scale(0.5)

        #self.play(FadeIn(vg_scene[0]))
        vg_scene[1].next_to(vg_scene[0], RIGHT)
        vg_scene[2].next_to(vg_scene[1], RIGHT)
        vg_scene[3].next_to(vg_scene[2], RIGHT)

        self.play(FadeIn(vg_scene))
        # self.play(ReplacementTransform(vg_scene[0], vg_scene[1]))
        self.wait(2)

        self.play(vg_scene.animate.shift(vg_scene[0].width*LEFT))

        self.wait(2)

