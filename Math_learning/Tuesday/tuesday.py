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
        self.vg_text = None

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

        text1 = Text("其中一个是男孩",color=GREEN).scale(0.7)
        en_text1 = Text("One of them is a boy",color=GREEN).scale(0.5).next_to(text1, RIGHT)
        vg_text1 = VGroup(text1,en_text1).scale(0.85).next_to(vg_answer, 0.5*UP)
        self.vg_text = vg_text1


        self.play(self.vg_problem.animate.to_edge(LEFT),
                  Write(vg_text1))
        self.wait(1)

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

        text2 = Text("第一个是男孩", color=GREEN).scale(0.7)
        en_text2 = Text("The first one is a boy", color=GREEN).scale(0.5).next_to(text2, RIGHT)
        vg_text2 = VGroup(text2, en_text2).scale(0.85).next_to(self.vg_answer, 0.5*UP)
        self.play(Transform(self.vg_text, vg_text2))
        self.wait(2)

    def problem_trans(self):
        answer1, t0, answer2, t1 = self.vg_answer

        cell = t1.get_entries((1, 2))
        self.play(FadeOut(cell))
        self.wait(2)

        answer3 = MathTex("P=\\frac{1}{2}").move_to(answer2)

        c = t1.get_columns()[0]
        r = t0.get_rows()[0]

        self.play(Circumscribe(c),
                  Circumscribe(r))
        self.wait(1)
        self.play(Transform(answer2, answer3))
        self.wait(1)

        self.play(Indicate(answer1),
                  Indicate(answer2))


class TuesdayProblem(Scene):
    def construct(self):
        self.vg_text = VGroup()
        self.ellipse = None
        self.vg_answer = None

        self.prob_contra()
        self.prob_anly2()
        self.prob_change()

    def prob_contra(self):
        text1 = Text("其中一个是男孩",color=BLUE).scale(0.7)
        en_text1 = Text("One of them is a boy", color=BLUE).scale(0.4).next_to(text1, 0.5*DOWN)
        self.vg_text.add(VGroup(text1,en_text1))

        text2 = Text("其中一个是男孩,出生在周二").scale(0.52)
        en_text2 = Text("One of them is a boy, born on Tuesday").scale(0.3).next_to(text2, 0.5*DOWN)
        self.vg_text.add(VGroup(text2, en_text2))

        text3 = Text("其中一个是男孩，出生在二月").scale(0.52)
        en_text3 = Text("One of them is a boy，born in February").scale(0.3).next_to(text3, 0.5*DOWN)
        self.vg_text.add(VGroup(text3, en_text3))

        text4 = Text("其中一个是男孩，出生在两点").scale(0.52)
        en_text4 = Text("One of them is a boy, born at two o'clock").scale(0.3).next_to(text4, 0.5*DOWN)
        self.vg_text.add(VGroup(text4, en_text4))

        text5 = Text("其中一个是男孩，出生1月2日").scale(0.52)
        en_text5 = Text("One of them is a boy, born on January 2nd ").scale(0.3).next_to(text5, 0.5*DOWN)
        self.vg_text.add(VGroup(text5, en_text5))

        text0 = Text('第一个是男孩', color=ORANGE).scale(0.7)
        en_text0 = Text("The first one is a boy", color=ORANGE).scale(0.4).next_to(text0, 0.5*DOWN)
        self.vg_text.add(VGroup(text0, en_text0))

        self.vg_text.arrange_submobjects(DOWN, buff=0.2).to_edge(LEFT)

        self.play(FadeIn(self.vg_text[0]), FadeIn(self.vg_text[-1]))
        #self.play(FadeIn(self.vg_text))

        ellipse1 = Ellipse(width=4, height=6, fill_opacity=0.8, color=BLUE)
        ellipse2 = Ellipse(width=2, height=3, fill_opacity=1, color=ORANGE)

        self.vg_ellipse = VGroup(ellipse1, ellipse2)

        self.play(FadeTransform(self.vg_text[0].copy(), ellipse1))
        self.play(FadeTransform(self.vg_text[-1].copy(), ellipse2))

        self.wait(2)

    def prob_anly2(self):

        svg_boy = SVGMobject('svg_icon/boy.svg', fill_color=BLUE).scale(0.55)
        svg_girl = SVGMobject('svg_icon/girl.svg', fill_color=RED).scale(0.55)
        t0 = MobjectTable([[svg_boy.copy(), svg_girl.copy()]])

        t1 = MobjectTable([[VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(RIGHT),
                            VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(LEFT)],
                           [VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                            VGroup()]])

        vg_answer = VGroup(t0, t1).arrange_in_grid(rows=2, buff=1).scale(0.6).to_edge(RIGHT)

        self.play(FadeTransform(self.vg_ellipse[0].copy(), t1))
        self.play(FadeTransform(self.vg_ellipse[1].copy(), t0))

        self.wait(1)
        self.vg_answer = vg_answer

    def prob_change(self):
        # 不确定坍缩
        arrow = Arrow(start=self.vg_text[0].get_edge_center(DOWN), end=self.vg_text[-1].get_edge_center(UP))
        self.play(GrowFromPoint(arrow, self.vg_text[0]))
        self.wait(1)

        self.play(self.vg_ellipse[0].animate.set(width=2, height=3, run_time=1))

        self.wait(1)
        cell = self.vg_answer[1].get_entries((1, 2))
        self.play(FadeOut(cell))

        self.wait(1)

class TuesdayAnaly(Scene):
    def construct(self):
        self.vg_table = None
        self.vg_problem = VGroup()

        self.intro_problem()
        self.problem_anly()
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


class ProblemGeneral(ThreeDScene):
    def construct(self):
        self.table = VGroup()
        self.vg_scene = None

        self.get_table()
        self.table_display()
        self.gener_anly()
        # self.get_func()

    @staticmethod
    def prob_anly(k):
        k = int(k)
        #k  = np.power(10,k)
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
        l_texts = [Text('男孩在周二出生'), Text('男孩在二月出生'), Text('男孩在两点出生'), Text('男孩在一月二日出生')]
        l_probs = [MathTex("\\frac{14-1}{28-1}", "=", "\\frac{13}{27}"),
                   MathTex("\\frac{24-1}{48-1}", "=", "\\frac{23}{47}"),
                   MathTex("\\frac{48-1}{96-1}","=","\\frac{47}{95}"),
                   MathTex("\\frac{730-1}{1460-1}","=","\\frac{729}{1459}")]
        for i in range(4):
            brace = Brace(vg_table[i], RIGHT)
            l_times[i].next_to(brace, RIGHT)
            l_texts[i].next_to(vg_table[i], UP)
            l_probs[i].scale(0.85).next_to(vg_table[i], DOWN)
            vg_scene.add(VGroup(vg_table[i], brace, l_times[i],l_texts[i],l_probs[i]).scale(0.85).to_edge(1.5*LEFT+0.8*UP))

        #vg_scene.arrange_submobjects(RIGHT, buff=1)
        #vg_scene.arrange_submobjects(RIGHT,buff=1).scale(0.9).shift(vg_scene[0].width*1.5*RIGHT)
        vg_scene.arrange_in_grid(rows=2, buff=0.8).scale(0.5)
        self.play(FadeIn(vg_scene))
        # vg_scene[1].next_to(vg_scene[0], RIGHT)
        # vg_scene[2].next_to(vg_scene[1], RIGHT)
        # vg_scene[3].next_to(vg_scene[2], RIGHT)

        self.move_camera(frame_center=vg_scene[0], zoom=2)
        # self.play(ReplacementTransform(vg_scene[0], vg_scene[1]))
        self.wait(2)
        self.move_camera(frame_center=vg_scene[1], zoom=2)
        self.wait(2)
        self.move_camera(frame_center=vg_scene[2], zoom=2)
        self.wait(2)
        self.move_camera(frame_center=vg_scene[3], zoom=2)
        self.wait(2)

        self.vg_scene = vg_scene

    def gener_anly(self):
        self.move_camera(frame_center=ORIGIN, zoom=1)

        vg_other = VGroup(*[vg[:4] for vg in self.vg_scene])
        vg_prob = VGroup(*[i[-1] for i in self.vg_scene])
        vg_res = VGroup(*[vg[-1] for vg in vg_prob])

        self.play(FadeOut(vg_other),
                  FadeOut(VGroup(*[vg[:2] for vg in vg_prob])),
                  vg_res.animate.arrange_submobjects(RIGHT, buff=0.5).scale(1.8).to_edge(UP))

        self.wait(2)

        math_inter = MathTex('\\frac{2k-1}{4k-1}', color=YELLOW).scale(0.9).to_edge(UP+LEFT)
        self.play(FadeIn(math_inter, target_position=vg_res))
        self.wait(1)

        ax = CommonFunc.add_axes(x_range=[0, 3], y_range=[0, 0.55], x_length=10, y_length=6,
                                 axis_config={"include_tip": True, "include_numbers": False},
                                 x_axis_config={"scaling": LogBase(custom_labels=False)}).scale(0.9).next_to(vg_res,DOWN)
        self.play(Create(ax))

        coords = [(7, 13/27), (12, 23/47), (24, 47/95), (360, 729/1459)]
        dns = [DecimalNumber(i, show_ellipsis=True, num_decimal_places=5) for i in [0.48148, 0.48936, 0.49473, 0.49965]]
        dnx = [Integer(i) for i in [7, 12, 24, 365]]
        for n, i in enumerate(coords):
            dot = Dot(ax.coords_to_point(i[0], i[1]), radius = 0.05, color=MAROON)
            v_line = ax.get_vertical_line(ax.c2p(i[0], i[1]))
            dns[n].next_to(dot, UP).scale(0.4)
            dnx[n].next_to(v_line, DOWN).scale(0.55)
            self.play(FadeTransform(vg_res[n], dot),
                      FadeIn(v_line),
                      Write(dnx[n]))
            self.play(Write(dns[n]))

        self.wait(1)
        fit_plot = ax.plot(lambda x: self.prob_anly(k=4*x), x_range=[0, 2.7], use_smoothing=True, color=YELLOW)
        fit_label = ax.get_graph_label(graph=fit_plot, label=MathTex('\\frac{2x-1}{4x-1}').scale(0.8)).next_to(fit_plot,LEFT)
        self.play(Create(fit_plot),
                  Write(fit_label))
        self.wait(1)

        h_line = ax.get_horizontal_line(ax.c2p(600, 0.51),line_func=DashedLine, color = RED)
        math_prob = MathTex('P=0.5', color=RED).scale(0.85).next_to(h_line, RIGHT)
        self.play(FadeIn(h_line), Write(math_prob))
        self.wait(1)

        new_dot = Dot(ax.coords_to_point(1, 1/3), radius=0.1, color=GREEN)
        new_tex = MathTex('(1,\\frac{1}{3})', color=GREEN).scale(0.6).next_to(new_dot, RIGHT)

        self.play(Create(new_dot),
                  Create(new_tex))
        self.wait(1)
        svg_boy = SVGMobject('svg_icon/boy.svg', fill_color=BLUE).scale(0.55)
        svg_girl = SVGMobject('svg_icon/girl.svg', fill_color=RED).scale(0.55)
        t1 = MobjectTable([[VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(RIGHT),
                            VGroup(svg_boy.copy(), svg_girl.copy()).arrange_submobjects(LEFT)],
                           [VGroup(svg_boy.copy(), svg_boy.copy()).arrange_submobjects(RIGHT),
                            VGroup()]]).scale(0.4).next_to(new_tex, DOWN)
        self.play(FadeIn(t1, target_mobject=new_dot))
        self.wait(1)

class TuesdayIntercept(Scene):
    def construct(self):
        self.vg_text = VGroup()
        self.ellipse = None
        self.vg_answer = None

        self.prob_contra()
        self.prob_ell()
        self.prob_path()
        #self.prob_anly2()
        #self.prob_change()

    def prob_contra(self):
        text1 = Text("其中一个是男孩", color=BLUE).scale(0.7)
        en_text1 = Text("One of them is a boy", color=BLUE).scale(0.4).next_to(text1, 0.5*DOWN)
        self.vg_text.add(VGroup(text1, en_text1))

        text2 = Text("其中一个是男孩，出生在周二").scale(0.52)
        en_text2 = Text("One of them is a boy, born on Tuesday").scale(0.3).next_to(text2, 0.5*DOWN)
        self.vg_text.add(VGroup(text2, en_text2))

        text3 = Text("其中一个是男孩，出生在二月").scale(0.52)
        en_text3 = Text("One of them is a boy，born in February").scale(0.3).next_to(text3, 0.5*DOWN)
        self.vg_text.add(VGroup(text3, en_text3))

        text4 = Text("其中一个是男孩，出生在两点").scale(0.52)
        en_text4 = Text("One of them is a boy, born at two o'clock").scale(0.3).next_to(text4, 0.5*DOWN)
        self.vg_text.add(VGroup(text4, en_text4))

        text5 = Text("其中一个是男孩，出生1月2日").scale(0.52)
        en_text5 = Text("One of them is a boy, born on January 2nd ").scale(0.3).next_to(text5, 0.5*DOWN)
        self.vg_text.add(VGroup(text5, en_text5))

        text0 = Text('第一个是男孩', color=ORANGE).scale(0.7)
        en_text0 = Text("The first one is a boy", color=ORANGE).scale(0.4).next_to(text0, 0.5*DOWN)
        self.vg_text.add(VGroup(text0, en_text0))

        self.vg_text.arrange_submobjects(DOWN, buff=0.2).to_edge(LEFT)

        self.play(FadeIn(self.vg_text[0]), FadeIn(self.vg_text[-1]))
        #self.play(FadeIn(self.vg_text))

        l_size = [(4, 6, 0.6, BLUE),
                  (3.6, 5.4, 0.75, GOLD),
                  (3.2, 4.8, 0.8, GREEN),
                  (2.8, 4.2, 0.85, YELLOW),
                  (2.4, 3.6, 0.9, TEAL),
                  (2, 3, 1, ORANGE)]
        vg_ellipse = VGroup(*[Ellipse(width=i[0], height=i[1], fill_opacity=i[2], color=i[-1]) for i in l_size]).next_to(self.vg_text, RIGHT)

        self.vg_ellipse = vg_ellipse
        self.wait(2)

    def prob_ell(self):
        self.play(Write(self.vg_text[1:5]))
        self.wait(1)

        for i in range(6):
            self.play(FadeTransform(self.vg_text[i].copy(), self.vg_ellipse[i]))
        self.wait(1)

    def prob_path(self):
        select_text = self.vg_text[1:5]
        coords = [(7, 13 / 27), (12, 23 / 47), (24, 47 / 95), (360, 729 / 1459)]
        l_size = [
                  (3.6, 5.4, 0.75, GOLD),
                  (3.2, 4.8, 0.8, GREEN),
                  (2.8, 4.2, 0.85, YELLOW),
                  (2.4, 3.6, 0.9, TEAL),
                  ]
        ax = CommonFunc.add_axes(x_range=[0, 3], y_range=[0, 0.55], x_length=5, y_length=3,
                                 axis_config={"include_tip": False, "include_numbers": False},
                                 x_axis_config={"scaling": LogBase(custom_labels=False)}).scale(0.9).next_to(self.vg_ellipse, 0.5*RIGHT)

        p = Variable(1/3, MathTex("P"), num_decimal_places=5).scale(0.9).next_to(ax, 4*UP)

        path = VMobject()
        dot = Dot(ax.c2p(1, 1/3), radius=DEFAULT_DOT_RADIUS*0.6, color=MAROON)
        path.set_points_as_corners([dot.get_center(), dot.get_center()])

        def update_path(path):
            previous_path = path.copy()
            previous_path.add_points_as_corners([dot.get_center()])
            path.become(previous_path)
        path.add_updater(update_path)

        self.play(FadeIn(ax),
                  Create(dot))
        self.add(path)
        self.play(Write(p))

        p.add_updater(lambda v: v.tracker.set_value(ax.point_to_coords(dot.get_center())[1]))

        for i in range(4):
            self.play(Indicate(select_text[i]),
                      self.vg_ellipse[i].animate.set(width=l_size[i][0], height=l_size[i][1]),
                      dot.animate.move_to(ax.c2p(coords[i][0], coords[i][-1])),run_time=1)
            self.play(FadeOut(self.vg_ellipse[i]))

        self.wait(2)

        h_line = ax.get_horizontal_line(ax.c2p(500, 0.52),line_func=DashedLine, color = RED)
        math_prob = MathTex('P=0.5', color=RED).scale(0.6).next_to(h_line, UP)

        self.play(FadeTransform(self.vg_ellipse[-1].copy(),VGroup(h_line, math_prob)))
        self.wait(1)

        self.play(self.vg_ellipse[-2].animate.set(width=2.1, height=3.1),
                  dot.animate.move_to(ax.c2p(600, 0.49999)), run_time=10)

    def prob_change(self):
        # 不确定坍缩
        arrow = Arrow(start=self.vg_text[0].get_edge_center(DOWN), end=self.vg_text[-1].get_edge_center(UP))
        self.play(GrowFromPoint(arrow, self.vg_text[0]))
        self.wait(1)

        self.play(self.vg_ellipse[0].animate.set(width=2, height=3, run_time=1))

        self.wait(1)
        cell = self.vg_answer[1].get_entries((1, 2))
        self.play(FadeOut(cell))

        self.wait(1)


