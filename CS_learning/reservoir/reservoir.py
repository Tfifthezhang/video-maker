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



class intro(Scene):
    def construct(self):
        eq = Tex('$\\frac{m}{m+1} \\times \\frac{m+1}{m+2} \\times \\frac{m+2}{m+3} '
                 '\\times ... \\times \\frac{n-2}{n-1}\\times \\frac{n-1}{n} = \\frac{m}{n}$').scale(1.5)
        text = Text('m和n都是正整数，并且有m<n').scale(0.7).to_edge(UP)
        text_en = Text('m and n are positive integers, and m<n', color=MAROON).scale(0.5).next_to(text, DOWN)
        self.play(Write(text))
        self.play(Write(text_en))

        self.wait(2)

        self.play(SpinInFromNothing(eq))

        self.wait(5)


class random_sampling(Scene):
    def construct(self):
        mod = 10
        axes = CommonFunc.add_bar_axes().scale(0.8).shift(LEFT)
        self.play(Write(axes))

        bars = CommonFunc.add_axes_bar(axes=axes, mod=mod)
        self.play(Create(bars))

        # self.play(bars[0].animate.stretch_to_fit_height(2).move_to(bars[0].get_bottom(), DOWN*0.1))

        count_tracker = ValueTracker(0)
        d_bar_height = dict.fromkeys(list(range(mod)), 0)

        labels = VGroup(*[Integer(0) for _ in range(mod)])

        def labels_update(labels):
            for i in range(len(labels)):
                labels[i].next_to(bars[i], UP * 0.2)
                labels[i].set_value(d_bar_height[i])

        labels.add_updater(labels_update)
        self.play(FadeIn(labels))

        C = Integer(0).next_to(axes, RIGHT)
        C.add_updater(lambda c: c.set_value(s))
        text = Text('0-9的随机数').scale(0.6).next_to(C, UP)
        self.add(text, C)
        for _ in range(520):
            s = random.randint(0, 9)
            bar = bars[s]
            if d_bar_height[s] <= 50:
                d_bar_height[s] = d_bar_height[s] + 1
                self.play(bar.animate.stretch_to_fit_height(0.1 * d_bar_height[s]).move_to(bar.get_bottom(), DOWN * 0.1))
            else:
                pass

        tex = Tex('$P=\\frac{1}{n}$',color=MAROON).scale(0.8).next_to(text, UP)
        self.play(Write(tex))
        self.wait(10)


class certainly_sampling(Scene):
    def construct(self):
        l_n = list(range(1, 101))
        random.shuffle(l_n)
        circle_number = CommonFunc.add_shape_object(l_n, rows=5, cols=20).scale(0.55).to_edge(UP)
        self.play(FadeIn(circle_number, lag_ratio=0.5))
        circles = circle_number[0]
        numbers = circle_number[1]

        brace_total = Brace(circle_number, DOWN, color=BLUE)
        tex_total = Tex('$n=100$').scale(0.6).next_to(brace_total, DOWN)
        self.play(Create(brace_total), Write(tex_total))

        self.wait(5)

        target_circles = VGroup(*[Circle().scale(0.55) for _ in range(10)]).scale(0.6)
        target_circles.arrange_submobjects(RIGHT, buff=SMALL_BUFF)
        target_circles.to_edge(4 * DOWN)
        sr = SurroundingRectangle(target_circles, buff=SMALL_BUFF)
        self.play(Create(sr))

        brace_sample = Brace(sr, DOWN, color=YELLOW)
        tex_sample = Tex('$m=10$').scale(0.6).next_to(brace_sample, DOWN)
        self.play(Create(brace_sample), Write(tex_sample))

        self.wait(5)

        random_value = CommonFunc.variable_tracker(label='s').scale(0.8).next_to(sr, UP + RIGHT)
        self.add(random_value)

        for i in range(10):
            s = random.randint(0, 99)
            number = numbers[s].copy()
            self.play(random_value.tracker.animate.set_value(s))
            self.play(number.animate.move_to(target_circles[i].get_center()))

        self.wait(5)


class reservoir_problem(Scene):
    def construct(self):
        self.slide_reservoir()
        self.insert_image()

    def slide_reservoir(self):
        l_n = list(range(1, 200))
        n = len(l_n)
        random.shuffle(l_n)
        circle_number = CommonFunc.add_shape_object(l_n, rows=1, cols=n, color=MAROON_A).scale(0.6)
        circles = circle_number[0]
        numbers = circle_number[1]

        circle_number.to_edge(LEFT, buff=SMALL_BUFF)
        self.play(Write(circle_number))

        stride = circles[0].width + SMALL_BUFF

        for i in range(100):
            self.play(circle_number.animate.shift(stride * LEFT))
            if i == 50:
                target_circles = VGroup(*[Circle().scale(0.55) for _ in range(10)]).scale(0.6)
                target_circles.arrange_submobjects(RIGHT, buff=SMALL_BUFF)
                target_circles.to_edge(2 * DOWN)
                sr = SurroundingRectangle(target_circles, buff=SMALL_BUFF)
                self.add(sr)

                self.play(FadeIn(target_circles), run_time=3)

                text1_cn = Text('当n很大或着未知，尤其是不能将n个元素全部加载到内存：').scale(0.6).to_edge(UP + LEFT)
                text1_en = Paragraph('The size of the population n is not known to the algorithm and',
                                     'is typically too large for all n items to fit into main memory').scale(0.4).next_to(
                    text1_cn, DOWN)

                text2_cn = Text('1.每个元素只能访问一次').scale(0.5).next_to(text1_en, DOWN).shift(3*LEFT)
                text2_en = Text('Each element can only be accessed once').scale(0.4).next_to(text2_cn)
                text3_cn = Text('2.每个元素采样概率相等').scale(0.5).next_to(text2_cn, DOWN)
                text3_en = Text('Each element has an equal probability of being sampled').scale(0.4).next_to(text3_cn)

                self.play(Write(text1_cn), Write(text1_en))
                self.wait(3)
                self.play(Write(text2_cn), Write(text2_en))
                self.wait(1)
                self.play(Write(text3_cn), Write(text3_en))


    def insert_image(self):
        image = SVGMobject('../../images/question.svg', fill_color=RED, color=BLUE).scale(0.8).shift(DOWN)
        self.play(SpinInFromNothing(image))



class reservoir_function(Scene):
    def construct(self):
        self.reservoir_method()

    def reservoir_method(self):
        l_n = list(range(1, 16))
        n = len(l_n)
        circle_number = CommonFunc.add_shape_object(l_n, rows=1, cols=n, color=MAROON_A).scale(0.7)
        circles = circle_number[0]
        numbers = circle_number[1]

        circle_number.to_edge(3 * UP)

        self.play(FadeIn(circle_number))

        self.wait(5)

        circles_m = circles[0:10]
        circles_m_out = circles[10:]
        numbers_m_out = numbers[10:]

        brace_sample = Brace(circles_m, DOWN, color=YELLOW)
        m_text = Tex('$m=10$').scale(0.8).next_to(brace_sample, DOWN)
        self.play(Create(brace_sample), Write(m_text))

        brace_total = Brace(circles, UP, color=BLUE)
        n_text = Tex('$n=15$').scale(0.8).next_to(brace_total, UP)
        self.play(Create(brace_total), Write(n_text))

        target_circles = VGroup(*[Circle().scale(0.55) for _ in range(10)]).scale(0.6)
        target_circles.arrange_submobjects(RIGHT, buff=SMALL_BUFF)
        target_circles.to_edge(3 * DOWN)
        sr = SurroundingRectangle(target_circles, buff=SMALL_BUFF)
        self.play(Create(sr))

        number_m = VGroup()
        for s in range(10):
            number = numbers[s].copy()
            number_m.add(number)
            self.play(number.animate.move_to(target_circles[s].get_center()))

        y = circles_m_out[0].get_center()[1]
        tracker = ValueTracker(circles_m_out[0].get_center()[0])
        pointer = Vector(UP).next_to(circles_m_out[0].get_center(), DOWN)
        pointer.add_updater(lambda m: m.next_to(np.array([tracker.get_value(), y, 0]), DOWN))
        self.add(pointer)

        random_value = CommonFunc.variable_tracker(label='random', start=0).scale(0.8).next_to(sr, UP)
        self.add(random_value)

        rands = [0, 10, 6, 12, 6]
        for i in range(5):
            number_new = numbers_m_out[i].copy()
            m_out_vg = VGroup(brace_sample, numbers_m_out[i])
            brace_m_out = Brace(m_out_vg, DOWN, color=MAROON_C, buff=brace_sample.height + m_text.height + SMALL_BUFF)
            text_m_out = Tex('$r={}$'.format(10 + 1 + i)).scale(0.8).next_to(brace_m_out, DOWN)

            self.play(tracker.animate.set_value(circles_m_out[i].get_center()[0]))
            self.play(Create(brace_m_out), Write(text_m_out))

            rand = rands[i]
            self.play(random_value.tracker.animate.set_value(rand))
            if rand < 10:
                rand_text = Tex('$<=9$').scale(0.9).next_to(random_value)
                self.play(FadeIn(rand_text))
                self.play(Indicate(number_m[rand]))
                self.play(FadeOut(number_m[rand]))
                number_m[rand] = number_new
                self.play(number_new.animate.move_to(target_circles[rand].get_center()))
                self.play(FadeOut(rand_text))
            else:
                pass
            self.play(FadeOut(brace_m_out), FadeOut(text_m_out))


class reservoir_solve(Scene):
    def construct(self):
        self.reservoir_explain()

    def reservoir_explain(self):
        l_n = list(range(1, 16))
        n = len(l_n)
        circle_number = CommonFunc.add_shape_object(l_n, rows=1, cols=n, color=MAROON_A).scale(0.7)
        circles = circle_number[0]
        numbers = circle_number[1]

        circle_number.to_edge(3 * UP)

        self.play(FadeIn(circle_number))

        circles_m = circles[0:10]
        circles_m_out = circles[10:]
        numbers_m_out = numbers[10:]

        brace_sample = Brace(circles_m, DOWN, color=YELLOW)
        m_text = Tex('$m=10$').scale(0.8).next_to(brace_sample, DOWN)
        self.play(Create(brace_sample), Write(m_text))

        brace_total = Brace(circles, UP, color=BLUE)
        n_text = Tex('$n=15$').scale(0.8).next_to(brace_total, UP)
        self.play(Create(brace_total), Write(n_text))

        target_circles = VGroup(*[Circle().scale(0.55) for _ in range(10)]).scale(0.6)
        target_circles.arrange_submobjects(RIGHT, buff=SMALL_BUFF)
        target_circles.to_edge(3 * DOWN)
        sr = SurroundingRectangle(target_circles, buff=SMALL_BUFF)
        self.play(Create(sr))

        number_m = VGroup()
        for s in range(10):
            number = numbers[s].copy()
            number_m.add(number)
            self.play(number.animate.move_to(target_circles[s].get_center()))

        self.play(FocusOn(circles[0]))
        P_vg = VGroup()
        P_text = Tex('$P(1)=1$').scale(0.8).next_to(sr, UP + LEFT)
        P_vg.add(P_text)
        self.add(P_text)
        self.play(Wiggle(P_text))

        y = circles_m_out[0].get_center()[1]
        tracker = ValueTracker(circles_m_out[0].get_center()[0])
        pointer = Vector(UP).next_to(circles_m_out[0].get_center(), DOWN)
        pointer.add_updater(lambda m: m.next_to(np.array([tracker.get_value(), y, 0]), DOWN))
        self.add(pointer)

        # rands = [4, 10, 6, 12, 6]
        for i in range(5):
            number_new = numbers_m_out[i].copy()
            m_out_vg = VGroup(brace_sample, numbers_m_out[i])
            brace_m_out = Brace(m_out_vg, DOWN, color=MAROON_C, buff=brace_sample.height + m_text.height + SMALL_BUFF)
            text_m_out = Tex('$r={}$'.format(10 + 1 + i)).scale(0.8).next_to(brace_m_out, DOWN)
            text_m_out_copy = text_m_out.copy()

            self.play(tracker.animate.set_value(circles_m_out[i].get_center()[0]))
            self.play(Create(brace_m_out), Write(text_m_out))

            new_frac = Tex('$\\times \\frac{%d}{%d}$' % (10 + i, 10 + i + 1)).scale(0.9).next_to(P_vg[-1])
            P_vg.add(new_frac)
            self.play(text_m_out_copy.animate.move_to(new_frac.get_center()))
            self.play(FadeOut(text_m_out_copy))
            self.play(FadeIn(new_frac))
            # P_text.become(MathTex('P='+'1'+'\\times'+'\\frac{{0}}{{1}}'.format(10+i, 10+i+1)), match_center=True)

            self.play(FadeOut(brace_m_out), FadeOut(text_m_out))

        sr_frac, sr_frac_text = CommonFunc.surround_box_and_text(P_vg[1:], Tex('$P(random \\neq 1)$'),
                                                                 position=UP)
        self.play(Create(sr_frac), Write(sr_frac_text))

        final_answer = Tex('$=\\frac{10}{15}$').scale(0.9).next_to(P_vg[-1])
        self.play(Write(final_answer))
        self.wait(5)

        self.play(FadeOut(sr_frac), FadeOut(sr_frac_text))

        retain_group = VGroup(*[Tex('$P({})=1$'.format(dx)).scale(0.8).next_to(sr, UP + LEFT) for dx in range(1, 11)])
        self.play(Succession(*[Transform(P_text, retain_group[i]) for i in range(len(retain_group))]))

        self.wait(5)

        P_out_text = Tex('$P(11) = $').scale(0.8).next_to(P_text, 2 * UP)
        self.play(GrowFromPoint(P_out_text, P_text.get_center()))

        frac_out = Tex('$\\frac{10}{11}$', color=RED).scale(0.9).next_to(P_out_text)
        self.play(Write(frac_out))
        self.play(Indicate(frac_out, run_time=3))

        sr_frac, sr_frac_text = CommonFunc.surround_box_and_text(frac_out, Tex('$P(random \\in [0,9])$'), position=UP)
        self.play(Create(sr_frac), Write(sr_frac_text))

        P_out_vg, out_final_answer = P_vg[2:].copy(), final_answer.copy()
        self.play(P_out_vg.animate.next_to(frac_out))
        self.play(out_final_answer.animate.next_to(P_out_vg))

        out_group = VGroup(*[Tex('$P({})=$'.format(dx)).scale(0.8).next_to(P_text, 2 * UP) for dx in range(11, 16)])
        for j in range(len(out_group) - 1):
            self.play(Transform(P_out_text, out_group[j + 1]))
            self.play(Transform(VGroup(frac_out, P_out_vg[:j + 1]),
                                Tex('$\\frac{%d}{%d}$' % (10, 11 + j + 1), color=RED).scale(0.9).next_to(P_out_text)))


class reservoir_general(Scene):
    def construct(self):
        self.generate_final()
        self.image_insert()

    def image_insert(self):
        paper = ImageMobject('../../images/reservoir.png').scale(1.35)
        self.play(SpinInFromNothing(paper))
        self.play(paper.animate.move_to(2 * LEFT))
        image = ImageMobject('../../images/jeffrey_vitter.jpg').to_edge(2 * RIGHT)
        self.play(FadeIn(image))
        self.play(Write(Text('Jeffrey Vitter').scale(0.6).next_to(image, DOWN)))

    def generate_final(self):
        l_n = list(range(1, 16))
        n = len(l_n)
        circle_number = CommonFunc.add_shape_object(l_n, rows=1, cols=n, color=MAROON_A).scale(0.7)
        circles = circle_number[0]
        numbers = circle_number[1]

        circle_number.to_edge(3 * UP)

        self.play(FadeIn(circle_number))

        circles_m = circles[0:10]

        brace_sample = Brace(circles_m, DOWN, color=YELLOW)
        m_text = Tex('$m=10$').scale(0.8).next_to(brace_sample, DOWN)
        self.play(Create(brace_sample), Write(m_text))

        brace_total = Brace(circles, UP, color=BLUE)
        n_text = Tex('$n=15$').scale(0.8).next_to(brace_total, UP)
        self.play(Create(brace_total), Write(n_text))

        # 推广到一般情况

        numbers[9].become(Tex('$...$'), match_center=True)
        m_text.become(Tex('$m$'), match_center=True)

        numbers[-1].become(Tex('$...$'), match_center=True)
        n_text.become(Tex('$n$'), match_center=True)

        P_text_1 = MathTex('P(i|i\\in[1,m])=',
                           '1',
                           '\\times \\frac{m}{m+1}',
                           '\\times \\frac{m+1}{m+2}',
                           '\\times \\frac{m+1}{m+2}',
                           '\\times \\frac{m+2}{m+3}',
                           '\\times ...',
                           '\\times \\frac{n-1}{n}',
                           '= \\frac{m}{n}').scale(0.6).to_edge(6 * DOWN)

        P_text_2 = MathTex('P(i|i\\in[m+1,n])=',
                           '\\frac{m}{m+1}',
                           '\\times \\frac{m+1}{m+2}',
                           '\\times \\frac{m+1}{m+2}',
                           '\\times \\frac{m+2}{m+3}',
                           '\\times ...',
                           '\\times \\frac{n-1}{n}',
                           '= \\frac{m}{n}').scale(0.6).next_to(P_text_1, DOWN)

        self.play(Write(P_text_1))
        text_1_group = VGroup(*[SurroundingRectangle(P_text_1[i], color=GREEN_A) for i in range(1, 8)])
        self.play(Create(text_1_group[0]))
        self.play(Succession(*[ReplacementTransform(text_1_group[0], text_1_group[i])
                               for i in range(len(text_1_group))]))

        self.play(FadeOut(text_1_group))

        self.play(Write(P_text_2))
        text_2_group = VGroup(*[SurroundingRectangle(P_text_2[i], color=GREEN_B) for i in range(1, 7)])
        self.play(Create(text_2_group[0]))
        self.play(Succession(*[ReplacementTransform(text_2_group[0], text_2_group[i])
                               for i in range(len(text_2_group))]))
        self.play(FadeOut(text_2_group))
        self.wait(5)


class reservoir_display(Scene):
    def construct(self):
        l_n = list(range(1, 100))
        n = len(l_n)
        random.shuffle(l_n)
        circle_number = CommonFunc.add_shape_object(l_n, rows=1, cols=n, color=MAROON_A).scale(0.6)
        circles = circle_number[0]
        numbers = circle_number[1]

        circle_number.to_edge(LEFT, buff=SMALL_BUFF)

        circle_number.shift(UP)
        self.play(Write(circle_number))

        stride = abs(numbers[0].get_center()[0]-numbers[1].get_center()[0])

        target_circles = VGroup(*[Circle().scale(0.55) for _ in range(10)]).scale(0.6)
        target_circles.arrange_submobjects(RIGHT, buff=SMALL_BUFF)
        target_circles.to_edge(2 * DOWN)
        sr = SurroundingRectangle(target_circles, buff=SMALL_BUFF)
        self.play(Create(sr))

        self.play(FadeIn(target_circles))

        number_m = VGroup()
        for s in range(10):
            number = numbers[s].copy()
            number_m.add(number)
            self.play(number.animate.move_to(target_circles[s].get_center()))

        # start_n = 10
        # for i in range(start_n):
        #     self.play(circle_number.animate.shift(stride * LEFT))

        Vec = Vector(UP).next_to(circles[10].get_center(), DOWN)
        self.add(Vec)

        random_value = CommonFunc.variable_tracker(label='random', start=0).scale(0.8).next_to(sr, UP)
        self.add(random_value)

        for i in range(10, 20):
            number_new = numbers[i].copy()
            rand = random.randint(0, i)
            self.play(random_value.tracker.animate.set_value(rand))
            if rand < 10:
                rand_text = Tex('$<=9$').scale(0.9).next_to(random_value)
                self.play(FadeIn(rand_text))
                self.play(Indicate(number_m[rand]))
                self.play(FadeOut(number_m[rand]))
                number_m[rand] = number_new
                self.play(number_new.animate.move_to(target_circles[rand].get_center()))
                self.play(FadeOut(rand_text))
            else:
                pass
            self.play(circle_number.animate.shift(stride * LEFT))


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
