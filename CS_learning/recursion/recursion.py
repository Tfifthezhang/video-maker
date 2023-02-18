# -*- coding: utf-8 -*-

# Copyright (C) 2021 GRGBanking All Rights Reserved

# @Time    : 2022/12/26 5:09 下午
# @Author  : 张暐（zhangwei）
# @File    : reservoir.py.py
# @Email   : zhangwei58@grgbanking.com
# @Software: PyCharm

from manim import *
import numpy as np
import sys

sys.path.append('..')

from CS_learning.common_func import CommonFunc
import random

FRAME_WIDTH = config.frame_width
FRAME_HEIGH = config.frame_height

FRAME_X_RADIUS = config.frame_x_radius
FRAME_Y_RADIUS = config.frame_y_radius


class logo(Scene):
    def construct(self):
        phanton = CommonFunc.add_function(lambda x: 0.3 * np.sin(5 * x), x_range=(-3, 3))
        start, end = phanton.get_start(), phanton.get_end()

        e_minus = CommonFunc.add_arrow(np.array([-5, 2, 0]), start,
                                       color=RED, max_tip_length_to_length_ratio=0.05)
        e_plus = CommonFunc.add_arrow(start, np.array([-5, -2, 0]),
                                      color=RED, max_tip_length_to_length_ratio=0.05)

        q_average = CommonFunc.add_arrow(np.array([5, 2, 0]), end,
                                         color=GREEN, max_tip_length_to_length_ratio=0.05)
        q = CommonFunc.add_arrow(end, np.array([5, -2, 0]),
                                 color=GREEN, max_tip_length_to_length_ratio=0.05)

        self.play(GrowArrow(e_minus))
        self.play(GrowArrow(e_plus))
        self.wait()
        self.play(Create(phanton))
        self.wait()
        self.play(GrowFromPoint(q_average, end))
        self.play(GrowFromPoint(q, end))

        group = VGroup(phanton, e_minus, e_plus, q_average, q)

        self.play(group.animate.scale(0.6))

        phanton_group = VGroup(*[CommonFunc.add_function(lambda x: 0.3 * np.sin(5 * x - np.pi * 0.5 * dx),
                                                         x_range=(-3, 3)).scale(0.6) for dx in range(41)])

        text_cn = Text('迷路的小画家', font='HiraginoSansGB-W3').scale(0.7).next_to(group, DOWN)

        self.play(FadeIn(text_cn), run_time=3)

        svg_image = SVGMobject('svg_icon/bird.svg', fill_color=MAROON).scale(1).next_to(phanton, 2 * UP)
        self.play(SpinInFromNothing(svg_image))

        self.play(Succession(*[Transform(phanton, phanton_group[i]) for i in range(len(phanton_group))], run_time=5))


class Title(Scene):
    def construct(self):
        svg_object = SVGMobject('svg_icon/book.svg', fill_color=BLUE)
        svg_group = VGroup(*[svg_object.copy() for _ in range(10)]).scale(0.4)
        svg_group.arrange_submobjects(RIGHT, buff=0.2).shift(1 * UP)

        brace = Brace(svg_group, direction=UP, color=MAROON)

        section_text = Text('基础算法优化').scale(0.9).next_to(brace, UP)

        self.play(Create(svg_group))
        self.wait(5)

        self.play(FadeIn(brace), Create(section_text))

        self.play(Indicate(svg_group[2], run_time=2))

        text = Text('递归算法    Recursion Algorithm ').scale(0.7).next_to(svg_group, DOWN * 3)

        self.play(GrowFromPoint(text, svg_group[0].get_center(), run_time=2))
        self.wait(3)

        subtext = Text('-- 如果你不明白，请参见"递归"').scale(0.5).next_to(text, 1.5 * DOWN)
        self.play(Write(subtext))


class recursion_example(Scene):
    def construct(self):
        circle_number, var, pointer, tracker, label = self.iteration_progress()
        for i in [var, pointer, tracker, label]:
            self.play(FadeOut(i))
        self.play(circle_number.animate.to_edge(UP))

        vg_text = self.write_tex(circle_number[1])
        recur_code = self.recursion_intro(vg_text)
        self.play(recur_code.animate.shift(4 * LEFT, 2 * UP))

        self.stack(recur_code, circle_number)

    def iteration_progress(self):
        l_n = list(range(1, 11))
        random.shuffle(l_n)
        circle_number = CommonFunc.add_shape_object(l_n).scale(0.8)
        self.play(FadeIn(circle_number, lag_ratio=0.5))
        numbers = circle_number[1]

        y = numbers[0].get_center()[1]
        pointer, tracker, label = CommonFunc.pointer_tracker(numbers, label_name='a', y=y, direction=UP,
                                                             position=DOWN)
        self.add(pointer, tracker, label)

        var = CommonFunc.variable_tracker(label=Tex('$S$'), color=GREEN).next_to(circle_number, UP)
        self.play(Create(var))

        plus_sign = MathTex('+').scale(0.8).next_to(var, 2 * RIGHT)

        s = 0
        for i in range(len(numbers)):
            self.play(tracker.animate.set_value(numbers[i].get_center()[0]))
            i_number = numbers[i].copy()
            i_value = numbers[i].get_value()
            s = s + i_value

            self.play(Write(plus_sign))
            self.play(i_number.animate.move_to(plus_sign.get_right()+0.5*RIGHT))
            self.play(FadeOut(i_number), FadeOut(plus_sign))

            self.play(var.tracker.animate.set_value(s))

        return circle_number, var, pointer, tracker, label

    def write_tex(self, m_object):
        n = len(m_object)
        string_text = ['S_{}'.format(i) for i in range(n)]
        y = m_object.get_center()[1]

        vg_text = VGroup()
        vg_brace = VGroup()
        for i in range(0, n):
            if i == 0:
                brace = Brace(m_object[:i + 1], DOWN, color=MAROON, buff=SMALL_BUFF)
            else:
                brace = Brace(m_object[:i + 1], DOWN, color=MAROON, buff=vg_brace.height + 0.3 + SMALL_BUFF)
            # text_start = MathTex(string_text[i-1]+'+a_{}'.format(i)).scale(0.6).next_to(brace, DOWN)
            text_end = MathTex(string_text[i]).scale(0.5).next_to(brace, 0.5 * DOWN)

            vg_text.add(text_end)
            vg_brace.add(brace)

            self.play(Write(brace))
            self.play(FadeIn(text_end))
            # self.play(TransformMatchingTex(text_start, text_end))

        self.wait(5)
        self.play(FadeOut(vg_brace))
        self.play(vg_text.animate.arrange_submobjects(RIGHT, buff=0.3).scale(2).shift(UP))

        return vg_text

    def recursion_intro(self, vg_text):
        n = len(vg_text)
        iteration_curves = VGroup(*[CurvedArrow(vg_text[i].get_corner(DOWN), vg_text[i + 1].get_corner(DOWN),
                                                radius=0.8, angle=TAU / 4,
                                                tip_length=0.1, color=RED) for i in range(n - 1)])
        self.play(Create(iteration_curves))

        iteration_text = Text('迭代法').scale(0.6).next_to(iteration_curves, DOWN)
        self.play(Write(iteration_text))

        iter_code = CommonFunc.add_code('recursion/iter_code.py', 'python').to_edge(DOWN)
        self.play(Create(iter_code))
        self.wait(3)

        recursion_curves = VGroup(*[CurvedArrow(vg_text[i].get_corner(UP), vg_text[i - 1].get_corner(UP),
                                                radius=0.8, angle=TAU / 4,
                                                tip_length=0.1, color=BLUE) for i in range(n - 1, 0, -1)])
        self.play(Write(recursion_curves))
        recursion_text = Text('递归法').scale(0.6).next_to(recursion_curves, UP)
        self.play(Write(recursion_text))
        recur_code = CommonFunc.add_code('recursion/recur_code.py', 'python').to_edge(DOWN)

        self.play(Transform(iter_code, recur_code, run_time=3))
        self.wait(3)

        vg_vanish = VGroup(*[recursion_text, recursion_curves, iter_code, iteration_text, iteration_curves])

        self.play(FadeOut(vg_vanish), FadeOut(vg_text))

        return recur_code

    def stack(self, recur_code, circle_number):
        numbers = circle_number[1]
        n = len(numbers)
        # y = numbers[0].get_center()[1]
        # pointer, tracker, label = CommonFunc.pointer_tracker(numbers, label_name='a', y=y, direction=UP,
        #                                                      position=DOWN)
        # self.add(pointer, tracker, label)

        # var = CommonFunc.variable_tracker(label=Tex('$S$'), color=GREEN).next_to(recur_code, UP)
        # self.play(Create(var))

        s = VGroup(*[RoundedRectangle(corner_radius=0.5, height=1.5) for i in range(10)])
        s.arrange_submobjects(UP, buff=0.2).scale(0.35).next_to(recur_code, 15 * RIGHT)

        vg_sum_text = VGroup()
        for i in range(n - 1, -1, -1):
            sum_number = numbers[:i + 1]
            self.play(Indicate(recur_code.code[-1]))
            self.play(Wiggle(sum_number))

            brace = Brace(sum_number, DOWN)
            sum_text = MathTex('S_{}'.format(i)).scale(0.6).next_to(brace, DOWN)

            self.play(FadeIn(brace), FadeIn(sum_text))

            sum_text_move = sum_text.copy()

            self.play(Create(s[-i - 1]))
            self.play(sum_text_move.animate.move_to(s[-i - 1].get_center()))
            vg_sum_text.add(sum_text_move)

            self.play(FadeOut(brace), FadeOut(sum_text))

        self.wait(3)

        # 递归关系
        stack_curves = VGroup(*[CurvedArrow(s[i].get_corner(RIGHT), s[i + 1].get_corner(RIGHT),
                                            radius=0.4, angle=TAU / 4,
                                            tip_length=0.1, color=RED) for i in range(n - 1)])
        vg_stack_text = VGroup()
        vg_stack_minus = VGroup()
        for stack in range(len(stack_curves)):
            self.play(FadeIn(stack_curves[stack]))
            relation_tex = MathTex('-', color=MAROON).scale(0.6).next_to(stack_curves[stack], RIGHT)
            self.play(Create(relation_tex))
            copy_number = numbers[-stack - 1].copy()
            self.play(copy_number.animate.move_to(relation_tex.get_center() + RIGHT))

            vg_stack_minus.add(relation_tex)
            vg_stack_text.add(copy_number)

        # 触发边界条件

        var = CommonFunc.variable_tracker(label=Tex('$S$'), color=GREEN).next_to(recur_code, 2 * RIGHT)
        self.play(Create(var))

        for re_index in range(n - 1, -1, -1):
            if re_index == n - 1:
                value = numbers[-re_index - 1].get_value()
                self.play(Indicate(recur_code.code[1:3]))
                self.play(FadeOut(s[re_index]))
                self.play(vg_sum_text[re_index].animate.move_to(var.get_center() + RIGHT))
                self.wait(2)
                self.play(FadeOut(vg_sum_text[re_index]))
            else:
                value = value + vg_stack_text[re_index].get_value()
                self.play(Indicate(recur_code.code[-1]))
                self.play(FadeOut(stack_curves[re_index]))
                self.play(FadeOut(vg_stack_minus[re_index]))
                self.play(FadeOut(s[re_index]))
                tex = Tex('$+$').scale(0.6).next_to(var, RIGHT)
                self.play(FadeIn(tex))
                self.play(vg_stack_text[re_index].animate.move_to(var.get_center() + 1.5 * RIGHT))
                self.wait(2)
                self.play(FadeOut(vg_stack_text[re_index]))
                self.play(FadeOut(tex))
                self.play(FadeOut(vg_sum_text[re_index]), FadeOut(vg_sum_text[re_index]))

            self.play(var.tracker.animate.set_value(value))


class recursion_des(Scene):
    def construct(self):
        text0_cn, text0_en = self.title()
        text1_cn, text1_en, vg_anim = self.attrs1(text0_en)
        text2_cn, text2_en = self.attrs2(text1_en, vg_anim)
        text3_cn, text3_en = self.attrs3(text2_en)

        self.play(text2_cn.animate.scale(1.5), text2_en.animate.scale(1.5))
        self.play(Circumscribe(VGroup(text2_cn, text2_en)))

    def title(self):
        text0_cn = Text('递归的要素', font='SIL-Hei-Med-Jian').scale(0.8).to_edge(4 * RIGHT + UP)
        text0_en = Text('Elements of recursion algorithm').scale(0.5).next_to(text0_cn, DOWN)

        self.play(Write(text0_cn), Write(text0_en))

        return text0_cn, text0_en

    def attrs_1_anim(self):
        l_n = list(range(1, 11))
        n = len(l_n)
        random.shuffle(l_n)
        circle_number = CommonFunc.add_shape_object(l_n).scale(0.8)
        m_object = circle_number[1]

        string_text = ['S_{}'.format(i) for i in range(n)]
        y = m_object.get_center()[1]

        vg_text = VGroup()
        vg_brace = VGroup()
        for i in range(0, n):
            if i == 0:
                brace = Brace(m_object[:i + 1], DOWN, color=MAROON, buff=SMALL_BUFF)
            else:
                brace = Brace(m_object[:i + 1], DOWN, color=MAROON, buff=vg_brace.height + 0.3 + SMALL_BUFF)
            text_end = MathTex(string_text[i]).scale(0.5).next_to(brace, 0.5 * DOWN)

            vg_text.add(text_end)
            vg_brace.add(brace)

        #     self.play(Write(brace))
        #     self.play(FadeIn(text_end))
        #
        # self.wait(5)
        # self.play(FadeOut(vg_brace))
        # self.play(vg_text.animate.arrange_submobjects(RIGHT, buff=0.3).scale(2).shift(UP))

        vg_total = VGroup(m_object, vg_text, vg_brace)

        return vg_total

    def attrs1(self, text0_en):
        text1_cn = Text('1. 将问题纵向拆分，缩减问题规模').scale(0.5).next_to(text0_en, 3 * DOWN)
        text1_en = Text('enumeration scope').scale(0.4).next_to(text1_cn, DOWN)

        vg_anim = self.attrs_1_anim().scale(0.7).to_edge(LEFT).shift(3 * UP)
        self.play(FadeIn(vg_anim))

        vec = Vector(UP).next_to(vg_anim, RIGHT)
        self.play(Write(vec))

        self.play(Write(text1_cn), Write(text1_en))

        self.play(Unwrite(vec))
        return text1_cn, text1_en, vg_anim

    def attrs2(self, text1_en, vg_anim):
        text2_cn = Text('2. 解决方法与问题规模无关').scale(0.5).next_to(text1_en, 3 * DOWN)
        text2_en = Text('enumeration scope').scale(0.4).next_to(text2_cn, DOWN)

        self.play(FadeOut(vg_anim[0]), FadeOut(vg_anim[-1]))
        vg_text = vg_anim[1]
        vg_text.arrange_submobjects(RIGHT, buff=0.2).scale(2).to_edge(LEFT)
        self.play(Create(vg_text))

        recursion_tex = MathTex('S_n = S_{n-1} + a_n', color=MAROON).scale(0.8).next_to(vg_text, 3 * UP)

        self.play(AddTextWordByWord(recursion_tex))

        n = len(vg_text)
        recursion_curves = VGroup(*[CurvedArrow(vg_text[i].get_corner(UP), vg_text[i - 1].get_corner(UP),
                                                radius=0.8, angle=TAU / 4,
                                                tip_length=0.1, color=BLUE) for i in range(n - 1, 0, -1)])
        self.play(Write(recursion_curves))

        self.play(Circumscribe(recursion_tex))

        self.play(Write(text2_cn), Write(text2_en))

        self.wait(5)

        self.play(FadeOut(recursion_curves), FadeOut(vg_text), FadeOut(recursion_tex))

        return text2_cn, text2_en

    def attrs3(self, text2_en):
        text3_cn = Text('3. 包含终止条件，终止条件即初始状态').scale(0.5).next_to(text2_en, 3 * DOWN)
        text3_en = Text('enumeration scope').scale(0.4).next_to(text3_cn, DOWN)

        s = VGroup()
        v_stack = RoundedRectangle(corner_radius=0.5, height=1.2, color=BLUE).scale(0.35).to_edge(6 * LEFT + 2 * DOWN)
        self.play(FadeIn(v_stack))
        s.add(v_stack)

        # s = VGroup(*[RoundedRectangle(corner_radius=0.5, height=1.2) for _ in range(1)])
        # s.arrange_submobjects(UP, buff=0.15).scale(0.35).to_edge(3*LEFT)

        for i in range(20):
            rr = RoundedRectangle(corner_radius=0.5, height=1.2, color=BLUE).scale(0.35).next_to(s[-1], UP, buff=0.01)
            self.play(FadeIn(rr))
            s.add(rr)

        # brace = Brace(s[:10], direction=LEFT, color=MAROON)
        # self.play(Write(brace))

        vec = Vector(RIGHT).next_to(s[9], RIGHT, buff=0.01)
        self.play(Create(vec))

        self.play(FadeOut(s[10:]))

        self.play(Write(text3_cn), Write(text3_en))

        self.play(Uncreate(vec))

        for i in range(9, -1, -1):
            self.play(Uncreate(s[i]))

        return text3_cn, text3_en


class recursion_core(Scene):
    def construct(self):
        self.fibonacci_example()
        self.maxvalue()

    def fibonacci_example(self):
        text = Text('斐波那契数列 Fibonacci Sequence').scale(0.8).to_edge(UP)
        self.play(Write(text))
        l_n = [1,1,2,3,5,8,13,21,34,55,89]
        n = len(l_n)
        circle_number = CommonFunc.add_shape_object(l_n, rows=1, cols=n, color=BLUE).scale(0.8)
        circles = circle_number[0]
        numbers = circle_number[1]

        self.play(Create(circle_number))

        tex = MathTex('Fib(n-1)+Fib(n-2)').scale(0.8).next_to(circle_number, UP)
        self.play(Write(tex))

        for i in range(n-1, 1, -1):
            sr = SurroundingRectangle(circles[i-2:i+1], buff=SMALL_BUFF, fill_opacity=0.4)
            n1_arrow = CommonFunc.add_curvearrow(circles[i-1].get_bottom(), circles[i].get_bottom(), radius=2)
            n2_arrow = CommonFunc.add_curvearrow(circles[i-2].get_bottom(), circles[i].get_bottom(), radius=2)

            self.play(Create(sr))

            self.play(Create(n1_arrow), Create(n2_arrow))

            self.wait(2)
            self.play(FadeOut(sr), FadeOut(n1_arrow), FadeOut(n2_arrow))

        self.wait(3)
        self.clear()

    def maxvalue(self):
        text = Text('一个数组中的最大值 the maximum value in an array').scale(0.8).to_edge(UP)
        self.play(Write(text))

        n = 11
        l_n = list(range(11, n+11))
        random.shuffle(l_n)
        circle_number = CommonFunc.add_shape_object(l_n, cols=n, buff=4*SMALL_BUFF).scale(0.8)
        circles = circle_number[0]
        numbers = circle_number[1]

        self.play(Create(circle_number))

        recursion_tex = MathTex('M_n = \max(M_{n-1},a_n)').scale(0.8).next_to(circle_number, UP)
        self.play(Write(recursion_tex))

        for i in range(n-1, 0, -1):
            sr_rest = SurroundingRectangle(circles[:i], buff=SMALL_BUFF)
            brace = Brace(sr_rest, DOWN)
            brace_text = Text('前{}个数字的最大值 Maximum value of the first {}'.format(i,i)).scale(0.5).next_to(brace, DOWN)

            sr = SurroundingRectangle(circles[i], color=RED, buff=SMALL_BUFF)

            self.play(Create(sr), Create(sr_rest), Create(brace), Create(brace_text))

            self.wait(2)
            self.play(FadeOut(sr), FadeOut(sr_rest), FadeOut(brace), FadeOut(brace_text))



class HanoiTower(Scene):
    def construct(self):
        self.diskes1 = VGroup()
        self.diskes2 = VGroup()
        self.diskes3 = VGroup()
        self.diskes = VGroup(self.diskes1, self.diskes2, self.diskes3)
        self.poles = None
        self.code = None


        self.drawPin()

        self.hanoi_intro()
        self.recursion_explain()
        self.recursion_move()

        self.setHanoi(5, 0, fill_opacity=0.5)
        self.solveHanoi(5, 0, 2)

    def hanoi_intro(self):
        self.setHanoi(5, 0)
        intro = Text('汉诺塔问题 Tower of Hanoi').scale(0.7).to_edge(UP+LEFT)

        claim1 = Text('每次只能移动一个圆盘 Only one disk may be moved at a time').scale(0.4).to_edge(2.5*UP+LEFT)
        claim2 = Text('大盘不能叠在小盘上面 No disk may be placed on top of a disk that is smaller than it').scale(0.4).to_edge(3.5*UP+LEFT)

        self.play(Write(intro))

        self.play(Write(claim1), Write(claim2))

        self.wait(3)

        self.play(Unwrite(intro), Unwrite(claim1), Unwrite(claim2))

        self.play(FadeOut(self.diskes[0]))
        self.clearHanoi()

    def recursion_explain(self):
        # 第一步，移动5个盘子
        self.move_explain(s=5)

        # 第二步，移动4个盘子
        self.move_explain(s=4)

        # 第三步，移动3个盘子
        self.move_explain(s=3)

        # 第四步，移动2个盘子
        self.move_explain(s=2)

        # 第五步，移动1个盘子
        self.move_explain(s=1)


    def recursion_move(self):
        text = MathTex('{{MultiMove}}({{n}}, {{a, b, c}})').scale(0.8).to_edge(UP)
        self.play(Write(text))
        self.play(Indicate(text.submobjects[-2], run_time=4))
        # first
        arrow1 = CommonFunc.add_arrow(self.poles[0].get_center(),
                                      self.poles[1].get_center(), color=GREY, buff=0.5)
        self.play(Write(arrow1))
        text1 = MathTex('MultiMove(n-1, a, c, b)').scale(0.6).next_to(arrow1, UP)
        self.play(ReplacementTransform(text.copy(), text1))

        # second
        cur_arrow = CommonFunc.add_arrow(self.poles[0].get_end(),
                                         self.poles[2].get_end(), color=GREY, buff=1)
        self.play(Write(cur_arrow))
        text3 = MathTex('SingleMove(a,c)').scale(0.6).next_to(cur_arrow, UP)
        self.play(Write(text3))

        # thrid
        arrow2 = CommonFunc.add_arrow(self.poles[1].get_center(),
                                      self.poles[2].get_center(), color=GREY, buff=0.5)
        self.play(Write(arrow2))
        text2 = MathTex('MultiMove(n-1, b, a, c)').scale(0.6).next_to(arrow2, UP)
        self.play(ReplacementTransform(text.copy(), text2))

        text_code = VGroup(text, text1, text3, text2)
        self.play(FadeOut(arrow1), FadeOut(arrow2), FadeOut(cur_arrow))

        code = CommonFunc.add_code('recursion/multimove.py', 'python').scale(1.2)

        self.play(text_code.animate.arrange_submobjects(DOWN, buff=0.3))
        self.wait(2)
        self.play(ReplacementTransform(text_code, code))

        self.code = code

        self.wait(3)
        self.play(code.animate.scale(0.4).to_edge(UP))

    def move_explain(self, s=5):
        cn_text = Text('如何移动 {} 个物品？'.format(s)).scale(0.6).to_edge(UP)
        en_text = Text('How to move {} items'.format(s)).scale(0.4).next_to(cn_text, DOWN)
        self.setHanoi(s, 0)
        self.play(Create(cn_text), Create(en_text))
        self.diskes[0].remove(*self.diskes[0][1:])
        self.wait(1)
        self.setHanoi(s-1, 1, color=TEAL)

        self.moveDisk(0, 2)

        self.diskes[1].remove(*self.diskes[1][:])
        self.wait(1)
        self.setHanoi(s, 2)

        self.play(FadeOut(self.diskes[2]))
        self.clearHanoi()
        self.play(Uncreate(cn_text), Uncreate(en_text))


    def clearHanoi(self):
        self.diskes1 = VGroup()
        self.diskes2 = VGroup()
        self.diskes3 = VGroup()
        self.diskes = VGroup(self.diskes1, self.diskes2, self.diskes3)

    def drawPin(self):
        with_line = Line(np.array([0, 2.5, 0]), np.array([0, -2.5, 0]), color=MAROON)

        left_line = with_line.copy().shift(4*LEFT)

        right_line = with_line.copy().shift(4*RIGHT)

        self.play(FadeIn(left_line, right_line, with_line))

        poles = VGroup(left_line, with_line, right_line)

        self.poles = poles

    def setHanoi(self, order, index_pole, color=GREEN, fill_opacity=1):
        for i in range(order):
            disk = RoundedRectangle(fill_opacity=fill_opacity, color=color, stroke_color=WHITE, stroke_width=0.8,
                                    width=3/5*(order - i), height=0.5, corner_radius=0.25)
            disk.move_to(self.poles[index_pole].get_end() + np.array([0, 0.25+i * 0.5, 0]))
            self.diskes[index_pole].add(disk)
        self.play(FadeIn(self.diskes[index_pole]))

    def moveDisk(self, src, dest):
        disk = self.diskes[src][-1]
        self.diskes[src].remove(disk)
        self.diskes[dest].add(disk)
        path = VGroup()
        vertices = [disk.get_center(),
                    self.poles[src].get_start(),
                    self.poles[dest].get_start(),
                    self.poles[dest].get_end()
                    + np.array([0, 0.25+(len(self.diskes[dest]) - 1) * 0.5, 0])]
        path.set_points_as_corners(vertices)
        self.play(MoveAlongPath(disk, path, run_time=2))

    def solveHanoi(self, order, src, dest):
        # 问题拆分
        temp = ({0, 1, 2} - {src, dest}).pop()
        if order == 1:
            self.play(Indicate(self.code.code[3]))
            self.moveDisk(src, dest)
        elif order == 2:
            self.play(Indicate(self.code.code[2]))
            self.moveDisk(src, temp)
            self.play(Indicate(self.code.code[3]))
            self.moveDisk(src, dest)
            self.play(Indicate(self.code.code[4]))
            self.moveDisk(temp, dest)

        else:
            self.play(Indicate(self.code.code[2]))
            self.solveHanoi(order - 1, src, temp)
            self.play(Indicate(self.code.code[3]))
            self.solveHanoi(1, src, dest)
            self.play(Indicate(self.code.code[4]))
            self.solveHanoi(order - 1, temp, dest)


class recursion_maxvalue(Scene):
    def construct(self):
        pass

    def maxvalue_des(self):
        n = 11
        l_n = list(range(0, n))
        random.shuffle(l_n)
        circle_number = CommonFunc.add_shape_object(l_n, cols=n).scale(0.6).to_edge(2 * UP)
        circles = circle_number[0]
        numbers = circle_number[1]

        self.play(Create(circle_number))

        code_two_max = CommonFunc.add_code('recursion/two_max.py', 'python').next_to(circle_number, 5 * DOWN + LEFT)

        self.play(Write(code_two_max))
        #
        # for i in range(n - 1, -1, -1):
        #     sr = SurroundingRectangle(circles[:i], buff=0.05, color=YELLOW)
        #     self.play(Write(sr))
        #     self.play(Unwrite(sr))

        recursion_tex = MathTex('M_n = \max(M_{n-1},a_n)').scale(0.8).next_to(circle_number, 2 * DOWN)
        self.play(Write(recursion_tex))

        # self.play(circle_number.animate.to_edge(UP))


class thanks_end(Scene):
    def construct(self):
        svg_image = SVGMobject('svg_icon/bird.svg', fill_color=MAROON).scale(1.5).shift(2 * UP)

        text_moto = Text('像小鸟一样努力').scale(0.4).next_to(svg_image, DOWN)

        text = Text('特别感谢', font='SIL-Hei-Med-Jian').next_to(svg_image, 4 * DOWN)

        self.play(SpinInFromNothing(svg_image))
        # self.play(FadeIn(text_moto))

        self.play(Create(text))

        # horizon_line = Line(text.get_center()+5*LEFT, text.get_center()+5*RIGHT).shift(5*DOWN)

        names = ['轩哥码题', '溪亭日暮']
        name_group = VGroup(*[Text(i) for i in names])
        name_group.arrange_submobjects(RIGHT, buff=1).scale(0.6).next_to(text, 5 * DOWN)

        self.play(FadeIn(name_group, lag_ratio=0.5, run_time=4))
        self.wait(2)
