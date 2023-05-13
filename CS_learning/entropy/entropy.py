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

sys.path.append('..')

from CS_learning.common_func import CommonFunc
import scipy.stats as stats

from sklearn.datasets import make_moons, make_blobs, make_classification

np.random.seed(0)


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
        encode_A = MathTex('A=1')
        encode_B = MathTex('B=01').next_to(encode_A, DOWN)
        encode_C = MathTex('C=00').next_to(encode_B, DOWN)
        encode_D = MathTex('D=10').next_to(encode_C, DOWN)

        vg_encode_event = VGroup(encode_A, encode_B, encode_C, encode_D)
        vg_encode_event.arrange_submobjects(DOWN).scale(0.8).to_edge(UP + RIGHT)

        self.play(ReplacementTransform(self.vg_encode_event, vg_encode_event))

        self.vg_encode_event = vg_encode_event

        self.wait(2)

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

        self.event_prob_log()
        self.event_self_information()
        self.event_chart()

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
                                                      label=MathTex('\log_2 (\\frac{1}{p})'), direction=UP).scale(0.7)

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

        information_tex_1 = MathTex('l(i) = \log_2(\\frac{1}{p_i})').to_edge(2 * UP + LEFT)

        self.info_tex = information_tex_1

        self.play(Write(information_tex_1))
        self.wait(2)

        information_tex_2 = MathTex('l(p_i) = -\log_2(p_i)').to_edge(2 * UP + LEFT)

        self.play(Transform(self.info_tex, information_tex_2))
        self.wait(2)

        information_tex_3 = MathTex('\mathrm{I}(p_i) = -\log_2(p_i)').to_edge(2 * UP + LEFT)

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
        entropy_tex_1 = MathTex('\sum_i p_iI(p_i)').scale(0.95).to_edge(2 * UP).shift(3.5*RIGHT)
        self.entropy_tex = entropy_tex_1
        self.play(Write(entropy_tex_1))
        self.wait(2)

        entropy_example = MathTex("\\frac{1}{2} \\times 1 ",
                                  "+", "\\frac{1}{4} \\times 2",
                                  "+", "\\frac{1}{8} \\times 3",
                                  "+", "\\frac{1}{8} \\times 3", "= 1.75").scale(0.65).next_to(self.vg_chart, DOWN)

        self.play(FadeTransform(self.entropy_tex.copy(), entropy_example))
        self.wait(2)

        entropy_tex_2 = MathTex("H(P) = \sum_i", "p_i", "I(p_i)").scale(0.95).to_edge(2 * UP).shift(3.5*RIGHT)
        self.play(Transform(self.entropy_tex, entropy_tex_2))
        self.wait(2)

        self.play(self.entropy_tex[1].animate.set_color(RED),
                  self.entropy_tex[2].animate.set_color(BLUE))
        self.wait(2)

        # entropy_tex_3 = MathTex("H(P) = \sum_i", "p_i", "[-\log_2 p_i]").scale(0.95).to_edge(2 * UP)
        # entropy_tex_3[1].animate.set_color(RED)
        # entropy_tex_3[2].animate.set_color(BLUE)
        # self.entropy_tex[2].animate.set_color(BLUE)
        # self.play(Transform(self.entropy_tex, entropy_tex_2))
        # self.wait(2)


class thanks_end(Scene):
    def construct(self):
        svg_image = SVGMobject('svg_icon/bird.svg', fill_color=MAROON).scale(1.5).shift(2 * UP)

        text = Text('感谢充电', font='SIL-Hei-Med-Jian').next_to(svg_image, 4 * DOWN)

        self.play(SpinInFromNothing(svg_image))

        self.play(Create(text))

        image1 = ImageMobject('svg_icon/charge/芽芽威武.jpg').scale(0.4).next_to(text, 4 * DOWN + LEFT)
        self.play(FadeIn(image1))
        name1 = Text('芽芽威武').scale(0.6).next_to(image1, RIGHT)
        self.play(FadeIn(name1))

        image2 = ImageMobject('svg_icon/charge/FuuLuu999.jpg').scale(0.29).next_to(name1, RIGHT)
        self.play(FadeIn(image2))
        name2 = Text('FuuLuu999').scale(0.6).next_to(image2, RIGHT)
        self.play(FadeIn(name2))

        self.wait(3)
