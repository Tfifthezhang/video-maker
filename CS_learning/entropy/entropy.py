# -*- coding: utf-8 -*-

# Copyright (C) 2021 GRGBanking All Rights Reserved

# @Time    : 2023/2/21 5:09 下午
# @Author  : 张暐（zhangwei）
# @File    : maximum_likelihood.py
# @Email   : zhangwei58@grgbanking.com
# @Software: PyCharm

from manim import *
import numpy as np
import sys
import os
import cv2

sys.path.append('..')

from CS_learning.common_func import CommonFunc
import scipy.stats as stats

from sklearn.datasets import make_moons, make_blobs, make_classification

np.random.seed(0)


class distribution_dis(Scene):
    def construct(self):
        self.vector_dis()
        self.dis_dis()

    def vector_dis(self):
        numberplane = NumberPlane(
            x_range=(-2, 5, 1),
            y_range=(-4.5, 4.5, 1),
            x_length=5,
            y_length=5,
        ).shift(3 * LEFT)
        self.play(Create(numberplane))
        rotation_center = 4 * LEFT
        theta_tracker = ValueTracker(110)
        line1 = Arrow(LEFT, RIGHT, color=GREEN, stroke_width=5, buff=0).shift(3 * LEFT)
        line_moving = Arrow(LEFT, RIGHT, color=RED, stroke_width=5, buff=0).shift(3 * LEFT)
        line_ref = line_moving.copy()
        line_moving.rotate(
            theta_tracker.get_value() * DEGREES, about_point=rotation_center
        )
        a = Angle(line1, line_moving, radius=0.5, other_angle=False)
        tex = MathTex(r"\theta").move_to(
            Angle(
                line1, line_moving, radius=0.5 + 3 * SMALL_BUFF, other_angle=False
            ).point_from_proportion(0.5)
        )

        self.play(Write(VGroup(line1, line_moving, a, tex)))
        self.wait(2)

        line_moving.add_updater(
            lambda x: x.become(line_ref.copy()).rotate(
                theta_tracker.get_value() * DEGREES, about_point=rotation_center
            )
        )

        a.add_updater(
            lambda x: x.become(Angle(line1, line_moving, radius=0.5, other_angle=False))
        )
        tex.add_updater(
            lambda x: x.move_to(
                Angle(
                    line1, line_moving, radius=0.5 + 3 * SMALL_BUFF, other_angle=False
                ).point_from_proportion(0.5)
            )
        )

        self.play(theta_tracker.animate.set_value(40))
        self.play(theta_tracker.animate.increment_value(140))
        self.play(tex.animate.set_color(RED), run_time=0.5)
        self.play(theta_tracker.animate.set_value(350))

        # vg_vector = VGroup(numberplane, line1, line_moving, a, tex, line_ref)
        # self.play(vg_vector.animate.to_edge(LEFT))
        # self.wait(1)

    def dis_dis(self):
        ax = CommonFunc.add_axes(x_range=[-6, 6], y_range=[0, 1], x_length=8, y_length=5.5,
                                 axis_config={"include_tip": False, "include_numbers": False}).scale(0.8).shift(3*RIGHT)
        self.play(Create(ax))

        graph_1 = ax.plot(lambda x: self.PDF_normal(x, 0, 1), x_range=[-5.9, 5.9], use_smoothing=True, color=RED)
        graph_2 = ax.plot(lambda x: self.PDF_normal(x, 1.5, 0.35), x_range=[-5.9, 5.9], use_smoothing=True, color=GREEN)

        self.play(Write(graph_1), Write(graph_2))

        for i in np.linspace(0.5, 2.5, 20):
            graph_other = ax.plot(lambda x: self.PDF_normal(x, mu=2-i, sigma=i), x_range=[-5.9, 5.9], use_smoothing=True,
                                  color=GREEN)
            self.play(Transform(graph_2, graph_other))

        self.wait(2)

    def PDF_normal(self, x, mu, sigma):
        return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


class SelfInfo(Scene):
    def construct(self):
        self.birds = VGroup()
        self.vg_01 = None
        self.vg_prob_event = None
        self.vg_encode_event = None
        self.encode_length = None

        self.trans_info_example()
        self.event_prob_example1()
        self.event_prob_example2()
        self.event_prob_example3()
        self.event_prob_example4()

    def trans_info_example(self):
        tree = RoundedRectangle(fill_opacity=0.7, color=GREEN, stroke_color=WHITE, stroke_width=0.1,
                                width=0.5, height=5, corner_radius=0.25).to_edge(DOWN)

        bird1 = SVGMobject('svg_icon/bird.svg', fill_color=GREY_BROWN).scale(1.5).next_to(tree, 6 * RIGHT)

        bird2 = SVGMobject('svg_icon/bird.svg', fill_color=GRAY_BROWN).scale(1.5).next_to(tree, 6 * LEFT)

        self.play(Create(tree))
        self.play(FadeIn(bird1), FadeIn(bird2))
        self.wait(2)

        self.birds = VGroup(bird2, tree, bird1)

        bird_chirp1 = Text('叽叽喳喳').scale(0.6).next_to(bird1, 0.2 * UP)
        bird_chirp2 = Text('叽叽喳喳').scale(0.6).next_to(bird2, 0.2 * UP)

        vg_text = VGroup(bird_chirp1, bird_chirp2)
        self.play(Write(vg_text))
        self.wait(2)

        text_bird_0 = Text("叽", color=BLUE).scale(0.7).to_edge(UP)
        text_bird_1 = Text("喳", color=RED).scale(0.7).next_to(text_bird_0)

        vg_bird_01 = VGroup(text_bird_0, text_bird_1)

        self.play(FadeTransform(vg_text, vg_bird_01))
        self.wait(2)

        tex_0 = MathTex("0", color=BLUE).scale(0.9).to_edge(UP)
        tex_1 = MathTex("1", color=RED).scale(0.9).next_to(tex_0)

        vg_tex = VGroup(tex_0, tex_1)

        self.play(ReplacementTransform(vg_bird_01, vg_tex))
        self.wait(2)

        self.vg_01 = vg_tex

    def event_prob_example1(self):
        # 概率事件
        prob_A = MathTex("P(A)", "=\\frac{1}{2}")
        prob_B = MathTex("P(B)", "=\\frac{1}{2}").next_to(prob_A, DOWN)

        vg_prob_event = VGroup(prob_A, prob_B)

        vg_prob_event.arrange_submobjects(DOWN).scale(0.6).to_edge(UP + LEFT)

        self.play(Create(vg_prob_event))

        self.vg_prob_event = vg_prob_event
        self.wait(2)

        # 对事件进行编码
        encode_A = MathTex("A=", "????")
        encode_B = MathTex("B=", "????").next_to(encode_A, DOWN)

        vg_encode_event = VGroup(encode_A, encode_B)
        vg_encode_event.arrange_submobjects(DOWN, buff=1).scale(0.8).to_edge(UP + 4 * RIGHT)

        self.play(FadeTransform(VGroup(self.vg_01.copy(), vg_prob_event.copy()), vg_encode_event))

        self.vg_encode_event = vg_encode_event
        self.wait(2)

        brace_A = Brace(encode_A[-1], color=RED, direction=DOWN)
        text_A = MathTex("l(A)").scale(0.7).next_to(brace_A)

        brace_B = Brace(encode_B[-1], color=RED, direction=DOWN)
        text_B = MathTex("l(B)").scale(0.7).next_to(brace_B)

        vg_length = VGroup(brace_A, text_A, brace_B, text_B)
        self.play(Write(vg_length))

        self.play(Circumscribe(self.vg_prob_event))
        self.wait(1)
        formula_target_1 = MathTex("\\frac{l(A)+l(B)}{2}").scale(0.8).next_to(self.vg_01, DOWN)
        self.play(FadeTransform(vg_length, formula_target_1))
        self.wait(2)

        formula_target_2 = MathTex("P(A)l(A)+P(B)l(B)").scale(0.8).next_to(self.vg_01, DOWN)
        self.play(FadeTransform(VGroup(formula_target_1, prob_A[0].copy(), prob_B[0].copy()), formula_target_2))
        self.wait(2)

        formula_target_3 = MathTex("\sum_i^2 P(i)l(i)").scale(0.8).next_to(self.vg_01, DOWN)
        self.play(ReplacementTransform(formula_target_2, formula_target_3))

        self.encode_length = formula_target_3

        for i, j in ([10000, 11001], [10000, 10001], [1100, 1001], [101, 111], [11, 10], [1, 0]):
            self.play(Transform(vg_encode_event[0][-1], MathTex(i).next_to(vg_encode_event[0][0])),
                      Transform(vg_encode_event[1][-1], MathTex(j).next_to(vg_encode_event[1][0])))

        self.wait(2)

        brace_bit = Brace(vg_encode_event[1][-1], color=MAROON, direction=DOWN)
        text_bit = MathTex('1 \\text{bit}').scale(0.7).next_to(brace_bit, DOWN)

        vg_bit = VGroup(brace_bit, text_bit)

        self.vg_encode_event.add(vg_bit)

        self.play(Create(vg_bit))
        self.wait(2)

        # 事件编码例子和传播

        event_emergence = MathTex('A').next_to(self.birds[0], 6 * UP)
        self.play(FadeIn(event_emergence))
        self.wait(2)

        encode_emergence = MathTex('1').next_to(self.birds[0], 0.2 * UP)
        mob = Circle(radius=6, color=TEAL_A).move_to(encode_emergence)
        self.play(FadeTransform(event_emergence, encode_emergence))
        self.wait(2)
        self.play(Broadcast(mob, focal_point=mob.get_center(), n_mobs=4))

        event_receive = MathTex('A').next_to(self.birds[-1], 0.2 * UP)
        self.play(FadeIn(event_receive))
        self.play(Flash(event_receive, flash_radius=0.25 + SMALL_BUFF))
        self.wait(2)

        self.play(FadeOut(encode_emergence), FadeOut(event_receive))

        # 例子2

        event_emergence = MathTex('B').next_to(self.birds[0], 6 * UP)
        self.play(FadeIn(event_emergence))
        self.wait(2)

        encode_emergence = MathTex('0').next_to(self.birds[0], 0.2 * UP)
        mob = Circle(radius=6, color=TEAL_A).move_to(encode_emergence)
        self.play(FadeTransform(event_emergence, encode_emergence))
        self.wait(2)
        self.play(Broadcast(mob, focal_point=mob.get_center(), n_mobs=4))

        event_receive = MathTex('B').next_to(self.birds[-1], 0.2 * UP)
        self.play(FadeIn(event_receive))
        self.play(Flash(event_receive, flash_radius=0.25 + SMALL_BUFF))
        self.wait(2)

        self.play(FadeOut(encode_emergence), FadeOut(event_receive))

    def event_prob_example2(self):
        # 概率事件
        prob_A = MathTex('P(A)=\\frac{1}{4}')
        prob_B = MathTex('P(B)=\\frac{1}{4}').next_to(prob_A, DOWN)
        prob_C = MathTex('P(C)=\\frac{1}{4}').next_to(prob_B, DOWN)
        prob_D = MathTex('P(D)=\\frac{1}{4}').next_to(prob_C, DOWN)

        vg_prob_event = VGroup(prob_A, prob_B, prob_C, prob_D)

        vg_prob_event.arrange_submobjects(DOWN).scale(0.6).to_edge(UP + LEFT)

        self.play(ReplacementTransform(self.vg_prob_event, vg_prob_event))

        self.vg_prob_event = vg_prob_event
        self.wait(2)

        # 对事件进行编码
        encode_A = MathTex('A=11')
        encode_B = MathTex('B=10').next_to(encode_A, DOWN)
        encode_C = MathTex('C=01').next_to(encode_B, DOWN)
        encode_D = MathTex("D=", "00").next_to(encode_C, DOWN)

        vg_encode_event = VGroup(encode_A, encode_B, encode_C, encode_D)
        vg_encode_event.arrange_submobjects(DOWN).scale(0.8).to_edge(UP + RIGHT)

        self.play(ReplacementTransform(self.vg_encode_event, vg_encode_event))

        self.vg_encode_event = vg_encode_event

        self.wait(2)

        brace_bit = Brace(vg_encode_event[3][-1], color=MAROON, direction=DOWN)
        text_bit = MathTex('2 \\text{bit}').scale(0.7).next_to(brace_bit, DOWN)

        vg_bit = VGroup(brace_bit, text_bit)

        self.vg_encode_event.add(vg_bit)

        self.play(Create(vg_bit))
        self.wait(2)

        # 让其平均长度最短
        formula_target = MathTex("\sum_i^4 P(i)l(i)=4\\times \\frac{1}{4} \\times 2").scale(0.8).next_to(self.vg_01,
                                                                                                         DOWN)
        self.play(Transform(self.encode_length, formula_target))
        self.wait(2)

        # 事件编码例子和传播

        event_emergence = MathTex('A').next_to(self.birds[0], 6 * UP)
        self.play(FadeIn(event_emergence))
        self.wait(2)

        encode_emergence = MathTex('11').next_to(self.birds[0], 0.2 * UP)
        mob = Circle(radius=6, color=TEAL_A).move_to(encode_emergence)
        self.play(FadeTransform(event_emergence, encode_emergence))
        self.wait(2)
        self.play(Broadcast(mob, focal_point=mob.get_center(), n_mobs=4))

        event_receive = MathTex('A').next_to(self.birds[-1], 0.2 * UP)
        self.play(FadeIn(event_receive))
        self.play(Flash(event_receive, flash_radius=0.25 + SMALL_BUFF))
        self.wait(2)

        self.play(FadeOut(encode_emergence), FadeOut(event_receive))

    def event_prob_example3(self):
        # 概率事件
        prob_A = MathTex('P(A)=\\frac{1}{8}')
        prob_B = MathTex('P(B)=\\frac{1}{8}').next_to(prob_A, DOWN)
        prob_C = MathTex('P(C)=\\frac{1}{8}').next_to(prob_B, DOWN)
        prob_D = MathTex('P(D)=\\frac{1}{8}').next_to(prob_C, DOWN)
        prob_E = MathTex('P(E)=\\frac{1}{8}').next_to(prob_D, DOWN)
        prob_F = MathTex('P(F)=\\frac{1}{8}').next_to(prob_E, DOWN)
        prob_G = MathTex('P(G)=\\frac{1}{8}').next_to(prob_F, DOWN)
        prob_H = MathTex('P(H)=\\frac{1}{8}').next_to(prob_G, DOWN)

        vg_prob_event = VGroup(prob_A, prob_B, prob_C, prob_D,
                               prob_E, prob_F, prob_G, prob_H)

        vg_prob_event.arrange_submobjects(DOWN).scale(0.6).to_edge(UP + LEFT)

        self.play(ReplacementTransform(self.vg_prob_event, vg_prob_event))

        self.vg_prob_event = vg_prob_event
        self.wait(2)

        # 对事件进行编码
        encode_A = MathTex('A=111')
        encode_B = MathTex('B=110').next_to(encode_A, DOWN)
        encode_C = MathTex('C=101').next_to(encode_B, DOWN)
        encode_D = MathTex('D=100').next_to(encode_C, DOWN)
        encode_E = MathTex('E=011').next_to(encode_D, DOWN)
        encode_F = MathTex('F=001').next_to(encode_E, DOWN)
        encode_G = MathTex('G=010').next_to(encode_F, DOWN)
        encode_H = MathTex("H=", "000").next_to(encode_G, DOWN)

        vg_encode_event = VGroup(encode_A, encode_B, encode_C, encode_D,
                                 encode_E, encode_F, encode_G, encode_H)
        vg_encode_event.arrange_submobjects(DOWN).scale(0.8).to_edge(UP + RIGHT)

        self.play(ReplacementTransform(self.vg_encode_event, vg_encode_event))

        self.vg_encode_event = vg_encode_event

        self.wait(2)

        brace_bit = Brace(vg_encode_event[7][-1], color=MAROON, direction=DOWN)
        text_bit = MathTex('3 \\text{bit}').scale(0.7).next_to(brace_bit, DOWN)

        vg_bit = VGroup(brace_bit, text_bit)

        self.vg_encode_event.add(vg_bit)

        self.play(Create(vg_bit))
        self.wait(2)

        # 平均编码长度最短
        formula_target = MathTex("\sum_i^8 P(i)l(i)=8\\times \\frac{1}{8} \\times 3").scale(0.8).next_to(self.vg_01,
                                                                                                         DOWN)
        self.play(Transform(self.encode_length, formula_target))
        self.wait(2)

        # 事件编码例子和传播

        event_emergence = MathTex('A').next_to(self.birds[0], 6 * UP)
        self.play(FadeIn(event_emergence))
        self.wait(2)

        encode_emergence = MathTex('111').next_to(self.birds[0], 0.2 * UP)
        mob = Circle(radius=6, color=TEAL_A).move_to(encode_emergence)
        self.play(FadeTransform(event_emergence, encode_emergence))
        self.wait(2)
        self.play(Broadcast(mob, focal_point=mob.get_center(), n_mobs=4))

        event_receive = MathTex('A').next_to(self.birds[-1], 0.2 * UP)
        self.play(FadeIn(event_receive))
        self.play(Flash(event_receive, flash_radius=0.25 + SMALL_BUFF))
        self.wait(2)

        self.play(FadeOut(encode_emergence), FadeOut(event_receive))

    def event_prob_example4(self):
        # 概率事件
        prob_A = MathTex('P(A)=\\frac{1}{2}')
        prob_B = MathTex('P(B)=\\frac{1}{4}').next_to(prob_A, DOWN)
        prob_C = MathTex('P(C)=\\frac{1}{8}').next_to(prob_B, DOWN)
        prob_D = MathTex('P(D)=\\frac{1}{8}').next_to(prob_C, DOWN)

        vg_prob_event = VGroup(prob_A, prob_B, prob_C, prob_D)

        vg_prob_event.arrange_submobjects(DOWN).scale(0.6).to_edge(UP + LEFT)

        self.play(ReplacementTransform(self.vg_prob_event, vg_prob_event))

        self.vg_prob_event = vg_prob_event
        self.wait(2)

        # 对事件进行编码示例1
        encode_A = MathTex('A=111')
        encode_B = MathTex('B=110').next_to(encode_A, DOWN)
        encode_C = MathTex('C=101').next_to(encode_B, DOWN)
        encode_D = MathTex('D=100').next_to(encode_C, DOWN)

        vg_encode_event = VGroup(encode_A, encode_B, encode_C, encode_D)
        vg_encode_event.arrange_submobjects(DOWN).scale(0.8).to_edge(UP + RIGHT)

        self.play(ReplacementTransform(self.vg_encode_event, vg_encode_event))

        self.vg_encode_event = vg_encode_event

        self.wait(2)

        # 平均编码长度最短
        formula_target = MathTex("\sum_i^4 P(i)l(i) = 3").scale(0.8).next_to(self.vg_01, DOWN)
        self.play(Transform(self.encode_length, formula_target))
        self.wait(2)

        # 对事件进行编码示例2
        encode_A = MathTex('A=11')
        encode_B = MathTex('B=10').next_to(encode_A, DOWN)
        encode_C = MathTex('C=01').next_to(encode_B, DOWN)
        encode_D = MathTex('D=00').next_to(encode_C, DOWN)

        vg_encode_event = VGroup(encode_A, encode_B, encode_C, encode_D)
        vg_encode_event.arrange_submobjects(DOWN).scale(0.8).to_edge(UP + RIGHT)

        self.play(ReplacementTransform(self.vg_encode_event, vg_encode_event))

        self.vg_encode_event = vg_encode_event

        self.wait(2)

        # 平均编码长度最短
        formula_target = MathTex("\sum_i^8 P(i)l(i) = 2").scale(0.8).next_to(self.vg_01, DOWN)
        self.play(Transform(self.encode_length, formula_target))
        self.wait(4)

        # 对事件进行编码示例3：不定长编码
        encode_A = MathTex("A=", "1")
        encode_B = MathTex('B=01').next_to(encode_A, DOWN)
        encode_C = MathTex('C=00').next_to(encode_B, DOWN)
        encode_D = MathTex("D=", "1", "0").next_to(encode_C, DOWN)

        vg_encode_event = VGroup(encode_A, encode_B, encode_C, encode_D)
        vg_encode_event.arrange_submobjects(DOWN).scale(0.8).to_edge(UP + RIGHT)

        self.play(ReplacementTransform(self.vg_encode_event, vg_encode_event))

        self.vg_encode_event = vg_encode_event

        self.wait(2)

        # 前缀说明
        text_1 = encode_A[1].copy()
        self.play(text_1.animate.move_to(encode_D[1]))
        self.play(Wiggle(encode_D[1:]))
        self.play(FadeOut(text_1))

        # 非事件编码例子和传播
        event_emergence = MathTex('A').next_to(self.birds[0], 6 * UP)
        self.play(FadeIn(event_emergence))
        self.wait(2)

        encode_emergence = MathTex('1').next_to(self.birds[0], 0.2 * UP)
        mob = Circle(radius=6, color=TEAL_A).move_to(encode_emergence)
        self.play(FadeTransform(event_emergence, encode_emergence))
        self.wait(2)
        self.play(Broadcast(mob, focal_point=mob.get_center(), n_mobs=4))

        event_receive = MathTex('A or D?').next_to(self.birds[-1], 0.2 * UP)
        self.play(FadeIn(event_receive))
        self.play(Wiggle(event_receive, run_time=1))
        self.wait(2)

        self.play(FadeOut(encode_emergence), FadeOut(event_receive))

        # 对事件进行编码示例4：不定长编码无公共前缀
        encode_A = MathTex('A=1')
        encode_B = MathTex('B=00').next_to(encode_A, DOWN)
        encode_C = MathTex('C=011').next_to(encode_B, DOWN)
        encode_D = MathTex('D=010').next_to(encode_C, DOWN)

        vg_encode_event = VGroup(encode_A, encode_B, encode_C, encode_D)
        vg_encode_event.arrange_submobjects(DOWN).scale(0.8).to_edge(UP + RIGHT)

        self.play(ReplacementTransform(self.vg_encode_event, vg_encode_event))

        self.vg_encode_event = vg_encode_event

        self.wait(2)

        # 平均编码长度最短
        formula_target = MathTex("\sum_i^8 P(i)l(i)=\\frac{1}{2} \\times 1 "
                                 "+ \\frac{1}{4} \\times 2 "
                                 "+ \\frac{1}{8} \\times 3 "
                                 "+ \\frac{1}{8} \\times 3  = 1.75").scale(0.8).next_to(self.vg_01, DOWN)
        self.play(Transform(self.encode_length, formula_target))
        self.wait(2)


class InfoEntropy(Scene):
    def construct(self):
        self.vg_chart = None
        self.vg_graph = None
        self.dot_lines = None
        self.info_tex = None
        self.entropy_tex = None
        self.entropy_example = None
        self.cross_entropy_tex = None

        self.event_prob_log()
        self.event_self_information()
        self.event_chart()
        self.cross_entropy()
        self.gibbs()

    def event_prob_log(self):
        # 分布直方图
        vg_chart = VGroup()
        for i, j in ([[0.5] * 2, ["A", "B"]],
                     [[0.25] * 4, ["A", "B", "C", "D"]],
                     [[0.125] * 8, ["A", "B", "C", "D", "E", "F", "G", "H"]]):
            # [[0.5, 0.25, 0.125, 0.125], ["A", "B", "C", "D"]]):
            barchart = BarChart(values=i,
                                bar_names=None,
                                y_range=[0, 1, 4],
                                y_length=2,
                                x_length=3,
                                x_axis_config={"font_size": 36}
                                )
            vg_chart.add(barchart)

        vg_chart.arrange_submobjects(RIGHT, buff=0.3).scale(0.6).to_edge(LEFT)

        self.vg_chart = vg_chart

        # 分布编码
        vg_info = VGroup()
        l_encoder = ['l(i)=1', 'l(i)=2', 'l(i)=3']
        l_p = ['P(i) = \\frac{1}{2}', 'P(i) = \\frac{1}{4}', 'P(i) = \\frac{1}{8}']
        for index_chart in range(len(l_encoder)):
            bit_tex = MathTex(l_encoder[index_chart]).scale(0.9).next_to(vg_chart[index_chart], UP)
            prob_tex = MathTex(l_p[index_chart]).scale(0.8).next_to(vg_chart[index_chart], DOWN)

            vg_info.add(VGroup(bit_tex, prob_tex))

        for k in range(len(vg_info)):
            self.play(Create(vg_chart[k]))
            self.play(Write(vg_info[k]))
            self.wait(1)

        vg_binary = VGroup(MathTex("2_{(10)}=", "10", "_{(2)}"),
                           MathTex("4_{(10)}=", "100", "_{(2)}"),
                           MathTex("8_{(10)}=", "1000", "_{(2)}"))
        for index_chart in range(len(l_encoder)):
            vg_binary[index_chart].scale(0.6).next_to(vg_info[index_chart], DOWN)

        l_len = [2, 3, 4]
        for k in range(len(vg_binary)):
            self.play(Write(vg_binary[k]))
            self.wait(2)
            brace = Brace(vg_binary[k][1], DOWN)
            brace_text = MathTex(str(l_len[k])).next_to(brace, DOWN)
            vg_binary.add(VGroup(brace, brace_text))
            self.play(Write(brace), Write(brace_text))
            self.wait(1)

        # binary digit 二进制编码，对数向下取整, 对数
        def bina_number(x):
            a = np.log2(x)
            if int(a) == a:
                return a
            else:
                return int(a) + 1

        ax = CommonFunc.add_axes(x_range=[0.01, 11.01], y_range=[-2, 5], x_length=6, y_length=2.5,
                                 axis_config={"include_tip": False, "include_numbers": True}).scale(1.1).to_edge(RIGHT)
        graph_prod = ax.plot(lambda x: bina_number(x), x_range=[0.11, 10.11], use_smoothing=False, color=BLUE)
        graph_label_prod = ax.get_graph_label(graph=graph_prod,
                                              label=MathTex('\lceil \log_2 (n) \\rceil'), direction=UP).scale(0.7)
        graph_log_prod = ax.plot(lambda x: np.log2(x), x_range=[0.11, 10], use_smoothing=True, color=MAROON)
        graph_label_log = ax.get_graph_label(graph=graph_log_prod,
                                             label=MathTex('\log_2 (n)'), direction=DOWN, color=MAROON).scale(0.7)

        self.play(Create(ax), Write(graph_label_prod))
        self.wait(2)

        self.play(FadeTransform(vg_binary, graph_prod))
        self.wait(3)
        self.play(FadeIn(graph_log_prod), Write(graph_label_log))
        self.wait(3)

        # log曲线

        log_ax = CommonFunc.add_axes(x_range=[0.01, 1.1], y_range=[0, 6], x_length=6, y_length=3,
                                     axis_config={"include_tip": False, "include_numbers": True}).scale(1.1).to_edge(
            RIGHT)
        log_graph_prod = log_ax.plot(lambda x: np.log2(1 / x), x_range=[0.02, 1], use_smoothing=True, color=BLUE)
        log_graph_label_prod = log_ax.get_graph_label(graph=log_graph_prod,
                                                      label=MathTex('\log_2 (\\frac{1}{p})'), direction=UP).scale(0.6)

        self.vg_graph = VGroup(log_ax, log_graph_prod, log_graph_label_prod)

        self.play(FadeTransform(VGroup(ax, graph_prod, graph_label_prod, graph_log_prod, graph_label_log),
                                self.vg_graph))
        self.wait(3)

        dot_examples = [[1 / 2, 1], [1 / 4, 2], [1 / 8, 3]]
        vg_dot = VGroup()

        for dot_xy in range(len(dot_examples)):
            dot = Dot(log_ax.coords_to_point(dot_examples[dot_xy][0], dot_examples[dot_xy][1]), color=MAROON)
            lines = log_ax.get_lines_to_point(log_ax.c2p(dot_examples[dot_xy][0], dot_examples[dot_xy][1]))
            print(lines)

            self.play(FadeTransform(vg_info[dot_xy], lines), FadeIn(dot))
            vg_dot.add(VGroup(dot, lines))
            self.wait(1)

        self.dot_lines = vg_dot

    def event_self_information(self):

        information_tex_1 = MathTex('l(i) = \log_2(\\frac{1}{p_i})').scale(0.8).to_edge(5 * UP + LEFT)

        self.info_tex = information_tex_1

        self.play(Write(information_tex_1))
        self.wait(2)

        information_tex_2 = MathTex('l(p_i) = -\log_2(p_i)').scale(0.8).to_edge(5 * UP + LEFT)

        self.play(Transform(self.info_tex, information_tex_2))
        self.wait(2)

        information_tex_3 = MathTex('\mathrm{I}(p_i) = -\log_2(p_i)').scale(0.8).to_edge(5 * UP + LEFT)

        self.play(Transform(self.info_tex, information_tex_3))

        self.wait(2)

        information_att_1 = MathTex('p(i,j) = p_i \\times p_j').next_to(self.vg_graph, 3.5 * UP)
        self.play(Write(information_att_1))
        self.wait(1)

        information_att_2 = MathTex('\mathrm{I}(p_{ij}) = -\log_2(p_i \\times p_j)').next_to(self.vg_graph, UP)
        self.play(FadeTransform(self.info_tex.copy(), information_att_2))
        self.wait(2)

        information_att_3 = MathTex('\mathrm{I}(p_{ij}) = -\log_2(p_i) + -\log_2(p_j)').next_to(self.vg_graph, UP)
        self.play(ReplacementTransform(information_att_2, information_att_3))
        self.wait(1)

        information_att_4 = MathTex('\mathrm{I}(p_{ij}) = \mathrm{I}(p_i) + \mathrm{I}(p_j)').next_to(self.vg_graph, UP)
        self.play(ReplacementTransform(information_att_3, information_att_4))
        self.wait(2)

        self.play(FadeOut(information_att_4),
                  FadeOut(information_att_1))
        self.wait(2)

        log_ax = self.vg_graph[0]
        dot = Dot(log_ax.coords_to_point(1, 0), color=RED)
        self.play(FadeIn(dot))
        self.vg_graph.add(dot)
        self.wait(3)

        self.play(FadeOut(dot))

    def event_chart(self):
        # 回到之前的不平等结果的分布
        barchart = BarChart(values=[0.5, 0.25, 0.125, 0.125],
                            bar_names=["A", "B", "C", "D"],
                            y_range=[0, 1, 4],
                            y_length=2,
                            x_length=3,
                            x_axis_config={"font_size": 36}
                            ).scale(0.9).next_to(self.vg_graph, 1.5 * LEFT)

        encode_A = MathTex('\mathrm{I}(A) = -\log(1/2) = 1')
        encode_B = MathTex('\mathrm{I}(B) = -\log(1/4) = 2').next_to(encode_A, DOWN)
        encode_C = MathTex('\mathrm{I}(C) = -\log(1/8) = 3').next_to(encode_B, DOWN)
        encode_D = MathTex('\mathrm{I}(D) = -\log(1/8) = 3').next_to(encode_C, DOWN)

        vg_encode_event = VGroup(encode_A, encode_B, encode_C, encode_D)
        vg_encode_event.arrange_submobjects(DOWN).scale(0.6).next_to(barchart, LEFT)

        self.play(Transform(self.vg_chart, barchart))
        self.wait(2)

        for encode in vg_encode_event:
            self.play(FadeTransform(VGroup(self.info_tex.copy(), barchart), encode))

        self.wait(2)

        self.vg_chart.add(vg_encode_event)

        # 信息熵
        entropy_tex_1 = MathTex('\sum_i p_iI(p_i)').scale(0.8).to_edge(2 * UP).shift(3.5 * RIGHT)
        self.entropy_tex = entropy_tex_1
        self.play(Write(entropy_tex_1))
        self.wait(2)

        # 信息熵代表着对概率分布P的最短平均编码长度，其本质是每一个事件分布的概率和其自信息是协调的，匹配的
        entropy_tex_2 = MathTex("H(P) = \sum_i", "p_i", "I(p_i)").scale(0.8).to_edge(2 * UP).shift(3.5 * RIGHT)
        self.play(Transform(self.entropy_tex, entropy_tex_2))
        self.wait(2)

        entropy_example = MathTex("H(P)=", "\\frac{1}{2} \\times 1 ",
                                  "+", "\\frac{1}{4} \\times 2",
                                  "+", "\\frac{1}{8} \\times 3",
                                  "+", "\\frac{1}{8} \\times 3", "= 1.75").scale(0.6).next_to(self.vg_chart, 2 * DOWN)

        self.play(FadeTransform(self.entropy_tex.copy(), entropy_example))
        self.wait(2)
        self.entropy_example = entropy_example

        # 自信息和概率协调
        self.play(self.entropy_tex[1].animate.set_color(RED),
                  self.entropy_tex[2].animate.set_color(BLUE))
        self.wait(2)

        log_ax = self.vg_graph[0]
        logp_graph_prod = log_ax.plot(lambda x: x, x_range=[0.02, 1], use_smoothing=True, color=RED)
        logp_graph_label_prod = log_ax.get_graph_label(graph=logp_graph_prod,
                                                       label=MathTex('p'), direction=UP + 7 * LEFT, color=RED).scale(
            0.65)

        vg_p_graph = VGroup(logp_graph_prod, logp_graph_label_prod)
        self.play(FadeTransform(self.entropy_tex[1].copy(), vg_p_graph))
        self.vg_graph.add(vg_p_graph)
        self.play(FadeOut(self.dot_lines))
        self.wait(1)

        # 开始迭代box，说明概率和对数概率协调更新
        vg_box = VGroup(*[SurroundingRectangle(self.entropy_example[i], buff=.05) for i in [1, 3, 5, 7]])
        t_labels = VGroup(
            *[log_ax.get_T_label(x_val=i, graph=self.vg_graph[1], label=MathTex(str(i))).scale(0.9) for i in
              [1 / 2, 1 / 4, 1 / 8, 1 / 8]])
        box = vg_box[0]
        t_label = t_labels[0]
        self.play(FadeIn(box), Write(t_label))

        for i in range(1, len(vg_box)):
            self.play(Transform(box, vg_box[i]),
                      Transform(t_label, t_labels[i]))
            self.wait(1)

        self.play(FadeOut(box), FadeOut(t_label))

        self.wait(2)

        self.play(FadeOut(self.vg_graph), FadeOut(vg_p_graph), FadeOut(self.entropy_example))

    def cross_entropy(self):
        # 事件产生
        vg_event = VGroup(*[MathTex(i) for i in 'ABCDABCD'])
        vg_event.arrange_submobjects(RIGHT, buff=0.2).scale(0.85).next_to(self.entropy_tex, 2.5 * DOWN)
        self.play(FadeIn(vg_event))
        self.wait(2)

        # 不同视角看待事件发生
        ## 从假设分布来看
        vg_info = VGroup()
        l_p = ['\\frac{1}{2}', '\\frac{1}{4}', '\\frac{1}{8}', '\\frac{1}{8}',
               '\\frac{1}{2}', '\\frac{1}{4}', '\\frac{1}{8}', '\\frac{1}{8}']
        for index_chart in range(len(l_p)):
            bit_tex = MathTex(l_p[index_chart]).scale(0.65).next_to(vg_event[index_chart], 2 * DOWN)
            vg_info.add(bit_tex)

        self.play(FadeTransform(self.vg_chart[0].copy(), vg_info))
        p_assume = MathTex('\\text{P}^{\\text{hypo}}', color=BLUE).scale(0.85).next_to(vg_info, LEFT)
        self.play(Write(p_assume))
        self.wait(2)

        ## 从观测分布来看
        vg_real = VGroup()
        l_p = ['\\frac{1}{4}', '\\frac{1}{4}', '\\frac{1}{4}', '\\frac{1}{4}',
               '\\frac{1}{4}', '\\frac{1}{4}', '\\frac{1}{4}', '\\frac{1}{4}']
        for index_chart in range(len(l_p)):
            bit_tex = MathTex(l_p[index_chart]).scale(0.65).next_to(vg_info[index_chart], 2 * DOWN)
            vg_real.add(bit_tex)

        self.play(FadeTransform(vg_event.copy(), vg_real))

        p_real = MathTex('\\text{P}^{\\text{real}}', color=RED).scale(0.85).next_to(vg_real, LEFT)
        self.play(Write(p_real))
        self.wait(2)

        ## 计算编码长度
        cross_entropy_1 = MathTex("\quad ", "\\times 1",
                                  "+", "\quad ", "\\times 2",
                                  "+", "\quad ", "\\times 3",
                                  "+", "\quad ", "\\times 3").scale(0.6).next_to(self.vg_chart, 2 * DOWN).shift(LEFT)

        for i in [1, 4, 7, 10]:
            cross_entropy_1[i][-1].set_color(BLUE)

        self.play(FadeTransform(self.vg_chart[-1].copy(), cross_entropy_1))
        self.wait(2)

        cross_entropy_2 = MathTex("\\frac{1}{4} ", "\\times 1",
                                  "+", "\\frac{1}{4} ", "\\times 2",
                                  "+", "\\frac{1}{4} ", "\\times 3",
                                  "+", "\\frac{1}{4} ", "\\times 3").scale(0.6).next_to(self.vg_chart, 2 * DOWN).shift(
            LEFT)

        for i in [1, 4, 7, 10]:
            cross_entropy_2[i][-1].set_color(BLUE)
        for j in [0, 3, 6, 9]:
            cross_entropy_2[j].set_color(RED)

        self.play(FadeTransform(VGroup(cross_entropy_1, p_real.copy(), vg_real.copy()), cross_entropy_2))
        self.wait(2)

        cross_entropy_3 = MathTex('H(\\text{P}^{\\text{real}},\\text{P}^{\\text{hypo}})').scale(0.6).next_to(
            cross_entropy_2, DOWN)
        self.play(FadeIn(cross_entropy_3))
        self.wait(2)

        # 引出交叉熵
        cross_entropy_4 = MathTex("H(\\text{P}^{\\text{real}},\\text{P}^{\\text{hypo}}) = ",
                                  "\sum_i", "p^{\\text{real}}_i", "\mathrm{I}(p_i^{\\text{hypo}})").scale(0.8).to_edge(
            2 * UP + LEFT)
        cross_entropy_4[2].set_color(RED)
        cross_entropy_4[3].set_color(BLUE)

        self.play(FadeTransform(cross_entropy_3, cross_entropy_4))
        self.wait(2)
        self.cross_entropy_tex = cross_entropy_4

        # 交叉熵要大于等于信息熵
        entropy_tex = MathTex("H(\\text{P}^{\\text{real}}) = ",
                              "\sum_i", "p^{\\text{real}}_i", "\mathrm{I}(p_i^{\\text{real}})").scale(0.8).to_edge(
            2 * UP + RIGHT)
        entropy_tex[2].set_color(RED)
        entropy_tex[3].set_color(RED)
        self.play(Transform(self.entropy_tex, entropy_tex))
        self.wait(2)

        morethan = MathTex("\geq").scale(0.9).next_to(self.cross_entropy_tex, 5 * RIGHT)
        self.play(FadeIn(morethan))
        self.wait(2)

        # 例证
        vg_info_2 = VGroup()
        l_p = ['\\frac{1}{4}', '\\frac{1}{4}', '\\frac{1}{4}', '\\frac{1}{4}',
               '\\frac{1}{4}', '\\frac{1}{4}', '\\frac{1}{4}', '\\frac{1}{4}']
        for index_chart in range(len(l_p)):
            bit_tex = MathTex(l_p[index_chart]).scale(0.65).next_to(vg_event[index_chart], 2 * DOWN)
            vg_info_2.add(bit_tex)

        self.play(FadeOut(vg_info), FadeOut(p_assume))
        self.play(FadeTransform(vg_real.copy(), vg_info_2))
        p_assume_2 = MathTex('\\text{P}^{\\text{real}}', color=RED).scale(0.85).next_to(vg_info_2, LEFT)
        self.play(Write(p_assume_2))
        self.wait(2)

        cross_entropy_base = MathTex("\\frac{1}{4} ", "\\times 2",
                                     "+", "\\frac{1}{4} ", "\\times 2",
                                     "+", "\\frac{1}{4} ", "\\times 2",
                                     "+", "\\frac{1}{4} ", "\\times 2", color=RED).scale(0.6).next_to(vg_real,
                                                                                                      2.5 * DOWN)
        self.play(FadeTransform(VGroup(vg_info_2.copy(), vg_real.copy()), cross_entropy_base))
        self.wait(1)

        cross_entropy_base_answer = MathTex('=2').scale(0.6).next_to(cross_entropy_base, RIGHT)
        cross_entropy_4 = MathTex('=2.25').scale(0.6).next_to(cross_entropy_2, RIGHT)
        self.play(Create(cross_entropy_4), Create(cross_entropy_base_answer))
        self.wait(2)

        self.play(FadeIn(MathTex('>').scale(0.9).next_to(cross_entropy_2).align_to(morethan, LEFT)))
        self.wait(2)

        self.entropy_example = VGroup(vg_event, vg_info_2, vg_real, p_assume_2, p_real)

    def gibbs(self):
        # 吉布斯不等式
        paper = ImageMobject('entropy/gibbs.png').scale(0.85).to_edge(0.75 * LEFT)
        self.play(SpinInFromNothing(paper))
        self.wait(2)

        # KL散度

        kl_dis = MathTex("H(\\text{P}^{\\text{real}},\\text{P}^{\\text{hypo}}) - H(\\text{P}^{\\text{real}})").scale(
            0.8).to_edge(RIGHT).shift(UP)

        brace = Brace(kl_dis, DOWN, color=YELLOW)
        brace_text = MathTex("D_{\mathrm{KL}}(\\text{P}^{\\text{real}}\| \\text{P}^{\\text{hypo}})",
                             color=GREEN).next_to(brace, DOWN)

        # kl_dis_2 = MathTex("= -\sum_i \\text{P}^{\\text{real}} \log_2 \\frac{\\text{P}^{\\text{hypo}}}"
        #                    "{\\text{P}^{\\text{real}}}").scale(0.75).next_to(kl_dis, DOWN).align_to(kl_dis[1], LEFT)
        #
        # kl_dis_3 = MathTex("= \sum_i \\text{P}^{\\text{real}} \log_2 \\frac{\\text{P}^{\\text{real}}}"
        #                    "{\\text{P}^{\\text{hypo}}}").scale(0.75).next_to(kl_dis_2, DOWN).align_to(kl_dis_2, LEFT)

        self.play(FadeOut(self.entropy_example))
        self.play(FadeTransform(VGroup(self.entropy_tex.copy(), self.cross_entropy_tex.copy()), kl_dis))
        self.wait(2)

        self.play(Create(brace))
        self.wait(1)
        self.play(FadeIn(brace_text))

        # self.play(Write(kl_dis_2))
        # self.play(Write(kl_dis_3))

        self.wait(2)


class regression(MovingCameraScene):
    def construct(self):
        self.dot_linspace = None
        self.dots = None
        self.ax = None
        self.mle_loss = None
        self.graph = None

        self.regression()
        self.regression_deduce()

    def regression(self):
        ax = CommonFunc.add_axes(x_range=[-8, 8], y_range=[-8, 8], x_length=8, y_length=6,
                                 axis_config={"include_tip": False, "include_numbers": False})
        self.play(Create(ax))
        self.ax = ax

        x = np.linspace(-7.5, 7.5, 150)
        gaussian_noise = np.random.normal(loc=0, scale=3, size=(150,))
        y = 0.8 * x - 0.7
        y_noise = y + gaussian_noise
        coords = list(zip(x, y_noise))

        dots = VGroup(
            *[Dot(ax.c2p(coord[0], coord[1]), radius=0.5 * DEFAULT_DOT_RADIUS, color=RED) for coord in coords])
        self.play(FadeIn(dots))

        self.dots = dots

        self.wait(1)

        self.camera.frame.save_state()

        dot = self.dots[100]
        self.play(self.camera.frame.animate.scale(0.3).move_to(dot))

        self.play(dot.animate.set(color=BLUE))

        iid_axes = Axes(x_range=[-3, 3], y_range=[0, 0.6], x_length=5, y_length=1,
                        axis_config=dict(include_tip=False,
                                         include_numbers=False,
                                         rotation=0 * DEGREES,
                                         stroke_width=1.0), ).scale(0.3).rotate(270 * DEGREES).next_to(dot,
                                                                                                       0.05 * RIGHT)
        self.play(Create(iid_axes))
        graph = iid_axes.plot(lambda x: self.normal_dis(x, mu=0, sigma=1),
                              x_range=[-3, 3],
                              use_smoothing=True,
                              color=BLUE)

        self.play(Create(graph))

        graph_noise = MathTex("y_i = wx_i + \epsilon_i").scale(0.3).next_to(graph, RIGHT)

        graph_noise_dis = MathTex("\epsilon_i", "\\thicksim", "\mathcal{N}(0,\sigma^2)", color=PURE_RED).scale(
            0.3).next_to(graph_noise, DOWN)
        self.play(Write(graph_noise), Write(graph_noise_dis))
        self.wait(2)

        graph_tex = MathTex("\mathcal{N}", "(", "y_i", "|", "\omega x_i", ",", "\sigma^2", ")").scale(0.4).next_to(
            graph, RIGHT)

        self.play(FadeTransform(VGroup(graph_noise, graph_noise_dis), graph_tex))
        self.graph = VGroup(graph, graph_tex)

        self.wait(3)
        self.play(Restore(self.camera.frame))
        self.wait(1)

        self.play(VGroup(self.ax, self.dots, graph_tex, iid_axes, graph).animate.shift(2 * RIGHT))
        self.wait(2)

    def regression_deduce(self):
        mle_loss_final = MathTex("\min_\omega \\frac{1}{n}", "\sum_i^n", "(y_i -\omega x_i)^2").scale(0.9).to_edge(
            1.5 * UP).shift(0.5 * LEFT)
        self.play(Write(mle_loss_final))
        self.wait(2)

        p_real = MathTex('\\text{P}^{\\text{real}}: \\frac{1}{n}', color=RED).scale(0.85).to_edge(LEFT + UP)

        self.play(FadeTransform(self.dots.copy(), p_real))
        self.wait(1)

        p_assume = MathTex("\\text{P}^{\\text{hypo}}:",
                           "\mathcal{N}", "(", "y_i", "|", "\omega x_i", ",", "\sigma^2", ")", color=BLUE).scale(
            0.85).next_to(p_real, DOWN).align_to(p_real, LEFT)

        self.play(FadeTransform(self.graph.copy(), p_assume))
        self.wait(1)

        normal_dis = MathTex(
            "p(y_i)= \\frac{1}{\sigma \sqrt{2 \pi}}",
            "e^{-\\frac{1}{2}",
            "\left(\\frac{y_i-\mu}{\sigma}", "\\right)^2}", color=BLUE).scale(0.65).next_to(p_assume, DOWN)

        self.play(FadeTransform(p_assume.copy(), normal_dis))
        self.wait(2)

        self.play(Indicate(normal_dis[2][-2], run_time=2))

        cross_entropy = MathTex("H", " = ",
                                "-\sum_i^n", "p^{\\text{real}}_i", "\log_2(p_i^{\\text{hypo}})").scale(0.8).next_to(
            normal_dis, 2 * DOWN).align_to(p_assume, LEFT)

        cross_entropy[3].set_color(RED)
        cross_entropy[4].set_color(BLUE)

        self.play(FadeIn(cross_entropy))
        self.wait(2)

        min_cross_entropy = MathTex(" = ", "-\sum_i^n", "\\frac{1}{n}",
                                    "-(y_i- \omega x_i)^2", "+f(\sigma)").next_to(cross_entropy, DOWN).scale(
            0.8).align_to(cross_entropy[1], LEFT)
        min_cross_entropy[2].set_color(RED)
        min_cross_entropy[3].set_color(BLUE)
        min_cross_entropy[4].set_color(BLUE)

        self.play(FadeTransform(VGroup(p_real.copy(), normal_dis.copy()), min_cross_entropy))
        self.wait(2)

        self.play(FadeOut(min_cross_entropy[-1]))
        self.wait(1)
        self.play(FadeOut(min_cross_entropy[3][0]), FadeOut(min_cross_entropy[1][0]))
        self.wait(2)

        self.play(min_cross_entropy[2].animate.move_to(min_cross_entropy[1][0]))
        self.play(min_cross_entropy[3][1:].animate.align_to(cross_entropy[3], LEFT))

        self.play(Circumscribe(mle_loss_final[1:], run_time=2),
                  Circumscribe(VGroup(min_cross_entropy[1][1:],
                                      min_cross_entropy[3][1:]), run_time=2))
        self.wait(1)

    def normal_dis(self, x, sigma, mu):
        coef = 1 / (sigma * np.sqrt(2 * np.pi))
        expon = -1 / 2 * ((x - mu) / sigma) ** 2
        return coef * np.power(np.e, expon)


class classification(ThreeDScene):
    def construct(self):
        self.sigmoid_ax = VGroup()
        self.class_ax = VGroup()
        self.bernoulli_graph = None

        self.make_01_classification()
        self.sigmoid_3D()
        self.formula_deduce()

    def make_01_classification(self):
        axes = ThreeDAxes(x_range=(-6, 6), y_range=(-6, 6), z_range=(-0.15, 1.15), x_length=7, y_length=7,
                          z_length=4).to_edge(2.5 * RIGHT)

        self.class_ax.add(axes)

        centers = [[-1, 1], [1, -1]]

        X, y = make_blobs(
            n_samples=100, centers=centers, cluster_std=0.45, random_state=0
        )
        coords = list(zip(X, y))

        dots = VGroup(
            *[Dot3D(axes.c2p(coord[0][0], coord[0][1], 0), radius=0.5 * DEFAULT_DOT_RADIUS, color=BLUE) if coord[
                                                                                                               1] == 0 else
              Dot3D(axes.c2p(coord[0][0], coord[0][1], 1), radius=0.5 * DEFAULT_DOT_RADIUS, color=RED) for coord in
              coords])

        self.class_ax.add(dots)

        self.play(Write(self.class_ax))
        self.wait(3)

    def sigmoid_3D(self):
        surface_1 = Surface(lambda u, v: self.class_ax[0].c2p(u, v, self.func_sigmoid(u, v)),
                            u_range=[-5, 5],
                            v_range=[-5, 5],
                            resolution=(30, 30),
                            should_make_jagged=True,
                            stroke_width=0.2,
                            )

        surface_1.set_style(fill_opacity=0.3)

        blues = [interpolate_color(BLUE, WHITE, i) for i in np.linspace(0, 1, 10)]
        reds = [interpolate_color(WHITE, RED, i) for i in np.linspace(0, 1, 10)]

        surface_1.set_fill_by_value(axes=self.class_ax[0], colorscale=blues + reds, axis=2)

        self.class_ax.add(surface_1)

        self.play(Create(surface_1))

        self.wait(2)

        self.play(Rotate(self.class_ax, angle=-90 * DEGREES, axis=RIGHT))
        self.wait(1)

        dot = self.class_ax[1][0]

        self.move_camera(frame_center=dot, zoom=2)

        inital_chart = BarChart(
            values=[0.5, 0.5],
            bar_names=["0", "1"],
            y_range=[0, 1, 10],
            y_length=4,
            x_length=2,
            x_axis_config={"font_size": 36},
            bar_colors=[WHITE, WHITE]
        ).scale(0.3).next_to(dot, 0.1 * UP)

        self.play(Create(inital_chart))
        self.wait(2)

        bernoulli_formula = MathTex('\mathcal{B}(p)').scale(0.4).next_to(inital_chart, RIGHT)
        self.play(FadeIn(bernoulli_formula))
        self.wait(1)

        self.move_camera(frame_center=ORIGIN, zoom=1)

        self.bernoulli_graph = VGroup(bernoulli_formula, inital_chart)

    def formula_deduce(self):
        mle_log = MathTex("\min -\\frac{1}{n}", "\sum_i^n[y_i \ln (p_i)+(1-y_i) \ln (1-p_i)]").scale(0.8).to_edge(
            1.5 * UP)
        self.play(Write(mle_log))
        self.wait(2)

        p_real = MathTex('\\text{P}^{\\text{real}}: \{ \\frac{n_1}{n}, \\frac{n_0}{n} \}', color=RED).scale(
            0.85).to_edge(LEFT + UP)

        self.play(FadeTransform(self.class_ax[1], p_real))
        self.wait(1)

        p_assume = MathTex("\\text{P}^{\\text{hypo}}:",
                           "\mathcal{B}(y_i | p_i)", color=BLUE).scale(0.85).next_to(p_real, DOWN).align_to(p_real,
                                                                                                            LEFT)

        self.play(FadeTransform(self.bernoulli_graph.copy(), p_assume))
        self.wait(1)

        normal_dis = MathTex("p_i", "^{y_i}(1-p_i)^{1-y_i}", color=BLUE).scale(0.65).next_to(p_assume, DOWN)

        self.play(FadeTransform(p_assume.copy(), normal_dis))
        self.wait(2)

        self.play(Indicate(normal_dis[0], run_time=2))

        cross_entropy = MathTex("H", " = ",
                                "-\sum_i^n", "p^{\\text{real}}_i", "\log_2(p_i^{\\text{hypo}})").scale(0.8).next_to(
            normal_dis, 2 * DOWN).align_to(p_assume, LEFT)

        cross_entropy[3].set_color(RED)
        cross_entropy[4].set_color(BLUE)

        self.play(FadeIn(cross_entropy))
        self.wait(2)

        min_cross_entropy = MathTex(" = ", "-", "\\frac{n_1}{n}", "\log_2(p_{y=1})",
                                    "-\\frac{n_0}{n}", "\log_2(1-p_{y=0})").next_to(cross_entropy, DOWN).scale(
            0.8).align_to(cross_entropy[1], LEFT)
        min_cross_entropy[2].set_color(RED)
        min_cross_entropy[4].set_color(RED)
        min_cross_entropy[3].set_color(BLUE)
        min_cross_entropy[5].set_color(BLUE)

        self.play(FadeTransform(VGroup(p_real.copy(), normal_dis.copy()), min_cross_entropy))
        self.wait(3)

        # self.play(CyclicReplace(min_cross_entropy[2], min_cross_entropy[3]))
        # self.wait(2)

        min_cross_entropy_2 = MathTex(" = ", "-", "\\frac{1}{n}", "\sum_i^n", "y_i", "\log_2(p_i)+",
                                      "(1-y_i)", "\log_2(1-p_i)").next_to(cross_entropy, DOWN).scale(0.8).align_to(
            cross_entropy[1], LEFT)

        min_cross_entropy_2[2].set_color(RED)
        min_cross_entropy_2[4].set_color(RED)
        min_cross_entropy_2[6].set_color(RED)
        min_cross_entropy_2[5].set_color(BLUE)
        min_cross_entropy_2[7].set_color(BLUE)

        self.play(ReplacementTransform(min_cross_entropy, min_cross_entropy_2))
        self.wait(2)

        # self.play(min_cross_entropy[2].animate.move_to(min_cross_entropy[0][0]))
        # self.play(min_cross_entropy[3][1:].animate.align_to(cross_entropy[3], LEFT))

        self.play(Circumscribe(mle_log[1][1:], run_time=2),
                  Circumscribe(min_cross_entropy_2[3:], run_time=2))
        self.wait(2)

    def func_sigmoid(self, x_1, x_2):
        s = np.power(np.e, -(x_1 - x_2))
        return 1 / (1 + s)

    def sigmoid(self, x):
        s = np.power(np.e, -x)
        return 1 / (1 + s)


class Cross_entropy(Scene):
    def construct(self):
        self.cross_entropy_explain()

    def cross_entropy_explain(self):
        ax = CommonFunc.add_axes(x_range=[0, 1.01], y_range=[0, 0.6], x_length=6, y_length=2.5,
                                 axis_config={"include_tip": False, "include_numbers": True}).scale(1.1).to_edge(RIGHT)
        graph_prod = ax.plot(lambda x: x * np.log2(1 / x), x_range=[0.001, 1], use_smoothing=True, color=MAROON)
        graph_label_prod = ax.get_graph_label(graph=graph_prod,
                                              label=MathTex('p\\times\log_2 (\\frac{1}{p})'), direction=8 * UP,
                                              color=MAROON).scale(0.7)

        vg_entropy_graph = VGroup(ax, graph_prod, graph_label_prod)

        self.play(Create(vg_entropy_graph))
        self.wait(2)


class entropy_3D(ThreeDScene):
    def construct(self):
        self.entropy_3D()

    def entropy_3D(self):
        axes = ThreeDAxes(x_range=(0, 1.01), y_range=(0, 1.01), z_range=(0, 0.6), x_length=7, y_length=7,
                          z_length=4)
        self.play(Create(axes))

        surface_1 = Surface(lambda u, v: axes.c2p(u, v, self.info_3D(u, v)),
                            u_range=[0.001, 1],
                            v_range=[0.001, 1],
                            resolution=(30, 30),
                            should_make_jagged=True,
                            stroke_width=0.2,
                            )

        surface_1.set_style(fill_opacity=0.3)

        blues = [interpolate_color(BLUE, WHITE, i) for i in np.linspace(0, 1, 10)]
        reds = [interpolate_color(WHITE, RED, i) for i in np.linspace(0, 1, 10)]

        surface_1.set_fill_by_value(axes=axes, colorscale=blues + reds, axis=2)

        self.play(Create(surface_1))

        vg_surface = VGroup(axes, surface_1)

        self.play(Rotate(vg_surface, angle=-90 * DEGREES, axis=RIGHT))
        self.wait(1)

        for i in range(6):
            self.play(Rotate(vg_surface, angle=60 * DEGREES, axis=UP))

        self.wait(2)

    def info_3D(self, x_1, x_2):
        s = x_1 * np.log2(1 / x_2)
        return s


class thanks_end(Scene):
    def construct(self):
        svg_image = SVGMobject('svg_icon/bird.svg', fill_color=MAROON).scale(1.2).to_edge(6 * LEFT + UP)

        text = Text('感谢充电', font='SIL-Hei-Med-Jian').scale(1.2).next_to(svg_image, 2.5 * DOWN)

        self.play(SpinInFromNothing(svg_image))

        self.play(Create(text))

        source_path = 'svg_icon/charge/'

        l_image_path = [image_path for image_path in os.listdir(source_path) if image_path.split('.')[-1] == 'jpg']
        vg_anchor = VGroup(*[Circle(1) for _ in range(len(l_image_path))])
        vg_anchor.arrange_in_grid(rows=3, buff=0.1).to_edge(2 * RIGHT)
        # self.add(vg_anchor)

        for i in range(len(l_image_path)):
            image_path = l_image_path[i]
            split_str = image_path.split('.')
            if split_str[-1] == 'jpg':
                name = split_str[0]
                # image = cv2.imread(os.path.join(source_path, image_path),3)
                # resize_array = cv2.resize(image, (512,512))
                image = ImageMobject(os.path.join(source_path, image_path)).move_to(vg_anchor[i].get_center())
                image.height = 1.2
                image.width = 1.2
                name_text = Text(name).scale(0.6).next_to(image, 0.5 * DOWN)
                self.add(image, name_text)

        self.wait(3)
