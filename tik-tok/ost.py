from manim import *
import numpy as np

import networkx as nx
from random import shuffle

from common_func import CommonFunc


class Title(Scene):
    pass


class HousePrice(Scene):
    def construct(self):
        self.trends()

    def trends(self):
        ax = CommonFunc.add_axes(x_range=[0, 6], y_range=[0, 100], x_length=6, y_length=4,
                                 axis_config={"include_tip": False, "include_numbers": False})

        svg_offer = SVGMobject('svg_icon/house_price.svg', fill_color=WHITE).scale(1)
        self.play(FadeIn(svg_offer))

        # 我有一所房子，挂牌一年，贬值过半
        path = VMobject()
        dot = Dot(ax.c2p(0, 100), radius=DEFAULT_DOT_RADIUS, color=RED)
        path.set_points_as_corners([dot.get_center(), dot.get_center()])

        def update_path(path):
            previous_path = path.copy()
            previous_path.add_points_as_corners([dot.get_center()])
            path.become(previous_path)

        path.add_updater(update_path)

        self.play(FadeIn(ax))
        self.play(ReplacementTransform(svg_offer, dot))

        self.wait(1)

        self.add(path)
        for x in np.linspace(0.1, 5, 10):
            self.play(dot.animate.move_to(ax.c2p(x, (x - 10) ** 2)))


class HouseDis(Scene):
    def construct(self):
        pass

    def price_dis(self):
        # 邻居抛售，拉低均价
        pass


class OST_BG(Scene):
    def construct(self):
        self.offer_group = None

        self.create_offer()
        self.offer_condition()

    def create_offer(self):
        svg_offer = SVGMobject('svg_icon/house_price.svg', fill_color=WHITE).scale(0.5)
        svg_people = SVGMobject('svg_icon/people.svg', fill_color=BLUE).scale(0.7).next_to(svg_offer, DOWN)
        vg_list = VGroup(svg_people, svg_offer)
        offer_group = VGroup(*[vg_list.copy() for _ in range(10)])
        offer_group.arrange_submobjects(RIGHT, buff=0.45).scale(0.8)
        self.play(Write(offer_group))
        self.wait(1)
        self.offer_group = offer_group

    def offer_condition(self):
        offer_group = self.offer_group
        # 满足三个条件
        ## 1. 我要尽可能选一个最高价，但我不知道哪一个是最高价
        for i in range(10):
            self.play(Flash(offer_group[i][-1],
                            color=YELLOW,
                            flash_radius=0.8))

        ## 2. 待选项个数是确定的

        brace = Brace(offer_group, direction=UP, color=YELLOW)
        brace_text = MathTex('N').scale(0.9).next_to(brace, UP)

        self.play(FadeIn(brace))
        self.play(Write(brace_text))

        ## 3. 我只能按照顺序一个一个谈，然后决定是否达成交易；
        arrow = Arrow(start=offer_group[0].get_left() + 1.5 * DOWN,
                      end=offer_group[-1].get_right() + 1.5 * DOWN,
                      color=MAROON, max_stroke_width_to_length_ratio=50)
        self.play(Write(arrow))
        for i in range(10):
            self.play(Indicate(offer_group[i], color=MAROON))
            self.play(offer_group[i].animate.set_color(GRAY))


class OST_anlyst(Scene):
    def construct(self):
        self.offer_group = None

        self.offer_create()
        self.group()
        self.process()
        self.policy_eq()
        self.anly()

    def offer_create(self):
        svg_offer = SVGMobject('svg_icon/house_price.svg', fill_color=WHITE).scale(0.5)
        svg_people = SVGMobject('svg_icon/people.svg', fill_color=BLUE).scale(0.7).next_to(svg_offer, DOWN)
        vg_list = VGroup(svg_people, svg_offer)
        offer_group = VGroup(*[vg_list.copy() for _ in range(10)])
        offer_group.arrange_submobjects(RIGHT, buff=0.45).scale(0.8).shift(0.5 * UP)
        self.add(offer_group)
        self.offer_group = offer_group

        brace = Brace(offer_group, direction=UP, color=YELLOW)
        brace_text = MathTex('N').scale(0.9).next_to(brace, UP)
        self.add(brace, brace_text)

        # 无论是选第一个，还是最后一个，最终的结果都是1/N
        vector = Vector(UP, color=BLUE).next_to(offer_group[0], DOWN)
        self.play(DrawBorderThenFill(vector))
        label = MathTex('P=\\frac{1}{N}', color=YELLOW).scale(0.8).next_to(vector, 0.5 * DOWN)
        self.play(FadeTransform(VGroup(brace, brace_text), label))
        label.add_updater(lambda m: m.next_to(vector, DOWN))
        self.wait(1)
        self.play(vector.animate.next_to(offer_group[-1], DOWN),
                  run_time=4)
        self.wait(1)

        self.play(FadeOut(vector),
                  FadeOut(label))

    def group(self):
        offer_group = self.offer_group

        line_ob = Line(start=offer_group[0].get_left(),
                       end=offer_group[3].get_right(),
                       stroke_width=20, color=RED)
        svg_offer = SVGMobject('svg_icon/house_price.svg', fill_color=WHITE).scale(0.4)
        svg_people = SVGMobject('svg_icon/people.svg', fill_color=WHITE).scale(0.6).next_to(svg_offer, DOWN)
        vg_list = VGroup(svg_people, svg_offer).next_to(line_ob, UP)
        vg_list.add_updater(lambda m: m.next_to(line_ob, UP))

        vg_ob = VGroup(line_ob, vg_list)

        self.play(ReplacementTransform(offer_group[:4], vg_ob))

        self.wait(1)

        line_select = Line(start=offer_group[4].get_left(),
                           end=offer_group[-1].get_right(),
                           stroke_width=20, color=BLUE)
        svg_offer = SVGMobject('svg_icon/house_price.svg', fill_color=WHITE).scale(0.4)
        svg_people = SVGMobject('svg_icon/people.svg', fill_color=WHITE).scale(0.6).next_to(svg_offer, DOWN)
        vg_list2 = VGroup(svg_people, svg_offer).next_to(line_select, UP)
        vg_list2.add_updater(lambda m: m.next_to(line_select, UP))

        vg_select = VGroup(line_select, vg_list2)

        self.play(ReplacementTransform(offer_group[4:], vg_select))

        self.wait(1)

        vg_line = VGroup(vg_ob, vg_select)

        self.vg_line = vg_line

        # brace = Brace(offer_group, direction=UP, color=YELLOW)
        # brace_text = MathTex('N').scale(0.9).next_to(brace, UP)
        #
        # self.add(brace, brace_text)

    def process(self):
        vg_line = self.vg_line

        ## 只看不选，获得其中的最大值
        vector_ob = Vector(UP).next_to(vg_line[0][0].get_left(), DOWN)
        self.play(DrawBorderThenFill(vector_ob))
        self.wait(1)

        var_value = [200, 215, 223, 210]
        var_position = (vg_line[0][0].get_right() - vg_line[0][0].get_left()) / 4

        ob_label = Variable(200, label=MathTex('o'), var_type=Integer).scale(0.8)
        ob_label.add_updater(lambda m: m.next_to(vector_ob, DOWN))

        res_value = VGroup(*[MathTex(str(i)) for i in var_value])
        res_value.arrange_submobjects(RIGHT, buff=0.3).next_to(vg_line[0], 3 * DOWN)

        self.play(FadeIn(ob_label))

        s = vg_line[0][0].get_left()
        for i in range(4):
            s = s + var_position
            self.play(vector_ob.animate.next_to(s, DOWN),
                      ob_label.tracker.animate.set_value(var_value[i]))
            self.play(FadeTransform(ob_label.copy(), res_value[i]))

        self.play(FadeOut(vector_ob),
                  FadeOut(ob_label))

        self.play(Indicate(res_value[2]))
        self.play(res_value[2].animate.set_color(RED),
                  FadeOut(VGroup(res_value[0], res_value[1], res_value[3])))

        ## 依次选取，遇见一个比之前的更好，就选它！

        vector_pick = Vector(UP).next_to(vg_line[1][0].get_left(), DOWN)
        self.play(DrawBorderThenFill(vector_pick))
        self.wait(1)

        pick_value = [205, 202, 210, 219, 199, 235]
        pick_position = (vg_line[1][0].get_right() / 2 - vg_line[1][0].get_left()) / 6

        pick_label = Variable(205, label=MathTex('s'), var_type=Integer).scale(0.9)
        pick_label.add_updater(lambda m: m.next_to(vector_pick, DOWN))

        self.play(FadeIn(pick_label))

        s = vg_line[1][0].get_left()
        for i in range(6):
            s = s + pick_position
            self.play(vector_pick.animate.next_to(s, DOWN),
                      pick_label.tracker.animate.set_value(pick_value[i]))
            self.play(Indicate(pick_label),
                      Indicate(res_value[2]))

        self.play(pick_label.animate.set_color(BLUE))
        self.play(FadeOut(vector_pick))

        ## 如果一直没有遇到，就选最后一个

        self.play(FadeOut(res_value[2]),
                  FadeOut(pick_label))

    def policy_eq(self):
        vg_line = self.vg_line

        left_gap = (vg_line[1][0].get_left() - vg_line[0][0].get_left()) * 9 / 10
        right_gap = (vg_line[1][0].get_right() - vg_line[0][0].get_right()) * 9 / 10

        # 观察区太小，选择区太大，就回到了选第一个的情况
        self.play(vg_line[0][0].animate.put_start_and_end_on(start=vg_line[0][0].get_left(),
                                                             end=vg_line[0][0].get_right() - left_gap),
                  vg_line[1][0].animate.put_start_and_end_on(start=vg_line[1][0].get_left() - left_gap,
                                                             end=vg_line[1][0].get_right()),
                  run_time=3)

        self.wait(1)

        # 观察区太大，选择区太小，就回到了选最后一个的情况
        self.play(vg_line[0][0].animate.put_start_and_end_on(start=vg_line[0][0].get_left(),
                                                             end=vg_line[0][0].get_right() + left_gap + right_gap),
                  vg_line[1][0].animate.put_start_and_end_on(start=vg_line[1][0].get_left() + left_gap + right_gap,
                                                             end=vg_line[1][0].get_right()),
                  run_time=5)

        self.wait(1)

        # 回到普通情况
        # self.play(vg_line[0][0].animate.put_start_and_end_on(start=vg_line[0][0].get_left(),
        #                                                      end=vg_line[0][0].get_right()-right_gap),
        #           vg_line[1][0].animate.put_start_and_end_on(start=vg_line[1][0].get_left()-right_gap,
        #                                                      end=vg_line[1][0].get_right()),
        #           run_time=2)
        #
        # self.wait(1)


class OST_MATH(Scene):
    def construct(self):
        self.vg_line = None
        self.vg_brace_tex = VGroup()

        self.create_line()
        self.anly()
        self.math_intro()

    def create_line(self):
        # 遇事不决，55开
        line_ob = Line(start=3 * LEFT, end=3.5 * RIGHT,
                       stroke_width=20, color=RED).to_edge(LEFT)
        svg_offer = SVGMobject('svg_icon/house_price.svg', fill_color=WHITE).scale(0.4)
        svg_people = SVGMobject('svg_icon/people.svg', fill_color=WHITE).scale(0.6).next_to(svg_offer, DOWN)
        vg_list = VGroup(svg_people, svg_offer).next_to(line_ob, UP)
        vg_list.add_updater(lambda m: m.next_to(line_ob, UP))

        vg_ob = VGroup(line_ob, vg_list)

        line_select = Line(start=3.5 * LEFT, end=3 * RIGHT,
                           stroke_width=20, color=BLUE).to_edge(RIGHT)
        svg_offer = SVGMobject('svg_icon/house_price.svg', fill_color=WHITE).scale(0.4)
        svg_people = SVGMobject('svg_icon/people.svg', fill_color=WHITE).scale(0.6).next_to(svg_offer, DOWN)
        vg_list2 = VGroup(svg_people, svg_offer).next_to(line_select, UP)
        vg_list2.add_updater(lambda m: m.next_to(line_select, UP))

        vg_select = VGroup(line_select, vg_list2)

        vg_line = VGroup(vg_ob, vg_select).shift(0.5 * UP)

        self.add(vg_line)
        self.vg_line = vg_line

    def anly(self):
        vg_line = self.vg_line
        # 假设观察区长度为a
        brace_ob = Brace(vg_line[0][0], direction=DOWN, color=WHITE)
        brace_ob.add_updater(lambda m: m.next_to(vg_line[0][0], DOWN))
        brace_text_ob = MathTex('a', color=RED).next_to(brace_ob, DOWN)
        brace_text_ob.add_updater(lambda m: m.next_to(brace_ob, DOWN))
        vg_brace_ob = VGroup(brace_ob, brace_text_ob)
        self.play(Write(vg_brace_ob))
        self.vg_brace_tex.add(vg_brace_ob)

        self.wait(1)

        # 最优的位置是在a+b处，因为我们要在选择区进行选择
        position_best = (vg_line[1][0].get_right() - vg_line[1][0].get_left()) / 2
        vector_ost = Vector(UP).next_to(vg_line[1][0].get_left() + position_best, 2 * DOWN)
        self.play(FadeIn(vector_ost))
        self.wait(1)

        line_b = Line(start=vg_line[1][0].get_left(),
                      end=vg_line[1][0].get_left() + position_best,
                      stroke_width=20, color=GREEN)
        line_other = Line(start=vg_line[1][0].get_left() + position_best,
                          end=vg_line[1][0].get_right(),
                          stroke_width=20, color=BLUE)

        vg_b = VGroup(line_b, line_other)

        self.play(ReplacementTransform(vg_line[1][0], vg_b))
        self.wait(1)

        brace_pick = Brace(vg_line[1][0][0], direction=DOWN, color=WHITE)
        brace_pick.add_updater(lambda m: m.next_to(vg_line[1][0][0], DOWN))
        brace_text_pick = MathTex('b', color=GREEN).next_to(brace_pick, DOWN)
        brace_text_pick.add_updater(lambda m: m.next_to(brace_pick, DOWN))
        vg_brace_pick = VGroup(brace_pick, brace_text_pick)

        self.play(Write(vg_brace_pick))

        self.wait(1)

        self.vg_brace_tex.add(vg_brace_pick)

        vector_ost.add_updater(lambda m: m.next_to(vg_line[1][0][0].get_right(), 2 * DOWN))

        ## 条件1：最优位置的概率是1/N

        prob_n = MathTex('P=', '\\frac{1}{N}', color=YELLOW).scale(0.8).next_to(vector_ost, DOWN)
        prob_n.add_updater(lambda m: m.next_to(vector_ost, DOWN))
        self.play(FadeIn(prob_n, target_position=vector_ost))

        vg_prob_vector = VGroup(vector_ost, prob_n)
        self.vg_brace_tex.add(vg_prob_vector)

        self.wait(2)

    def math_intro(self):
        vg_line = self.vg_line
        vg_brace_tex = self.vg_brace_tex
        ## 条件2：a+b区间的最大的必须要在a区间，概率为a/(a+b)

        self.play(Wiggle(vg_line[0][0], n_wiggles=3, rotate_about_point=ORIGIN, run_time=5),
                  Wiggle(vg_line[1][0][0], n_wiggles=3, rotate_about_point=ORIGIN, run_time=5))
        self.wait(1)
        self.play(ApplyWave(vg_line[0][0], ripples=2))
        self.wait(1)

        prob_ob = MathTex('\\frac{a}{a+b}').scale(0.9).shift(1.5 * UP)
        self.play(FadeTransform(VGroup(vg_line[0][0], vg_line[1][0][0]).copy(), prob_ob))
        self.wait(1)

        self.play(FadeOut(vg_brace_tex[1][0]))
        vg_brace_tex[1][1].add_updater(lambda m: m.next_to(vg_line[1][0][0], DOWN))

        prob_sum = MathTex('\sum').next_to(prob_ob, 0.5 * LEFT).shift(0.1 * UP)
        prob_sum_down = MathTex('b=0').scale(0.6).next_to(prob_sum, 0.1 * DOWN)
        prob_sum_up = MathTex('N-a-1').scale(0.6).next_to(prob_sum, 0.1 * UP)

        vg_prob_sum = VGroup(prob_sum_up, prob_sum, prob_sum_down)
        # self.play(FadeIn(vg_prob_sum))

        ## 并对所有可能的位置进行概率求和
        left_gap = (vg_line[1][0][1].get_left() - vg_line[1][0][0].get_left()) * 9 / 10
        right_gap = (vg_line[1][0][1].get_right() - vg_line[1][0][0].get_right()) * 9 / 10
        ### b最小可以到零
        self.play(vg_line[1][0][0].animate.put_start_and_end_on(start=vg_line[1][0][0].get_left(),
                                                                end=vg_line[1][0][0].get_right() - left_gap),
                  vg_line[1][0][1].animate.put_start_and_end_on(start=vg_line[1][0][1].get_left() - left_gap,
                                                                end=vg_line[1][0][1].get_right()),
                  run_time=3)

        min_value = MathTex('=0', color=WHITE).next_to(vg_brace_tex[1][1], 0.1 * RIGHT)
        self.play(FadeIn(min_value, target_position=vg_line[1][0][0]))

        self.wait(1)
        self.play(FadeTransform(min_value, vg_prob_sum[-1]))

        ### b最大可以到N-a-1
        self.play(vg_line[1][0][0].animate.put_start_and_end_on(start=vg_line[1][0][0].get_left(),
                                                                end=vg_line[1][0][
                                                                        0].get_right() + left_gap + right_gap),
                  vg_line[1][0][1].animate.put_start_and_end_on(
                      start=vg_line[1][0][1].get_left() + left_gap + right_gap,
                      end=vg_line[1][0][1].get_right()),
                  run_time=3)
        max_value = MathTex('=N-a-1', color=WHITE).next_to(vg_brace_tex[1][1], 0.1 * RIGHT)
        self.play(FadeIn(max_value, target_position=vg_line[1][0][0]))
        self.wait(1)
        self.play(FadeTransform(max_value, vg_prob_sum[0]))

        self.wait(1)
        self.play(FadeIn(prob_sum[0]))
        self.wait(1)

        # 最后需要乘上1/N

        prob_N = MathTex('\\frac{1}{N}', color=YELLOW).next_to(vg_prob_sum[1], LEFT)
        self.play(FadeOut(vg_brace_tex[-1][-1], target_position=prob_N),
                  FadeIn(prob_N))
        self.wait(1)


class OST(ThreeDScene):
    def construct(self):
        self.vg_tex = None
        self.vg_line = None
        self.vg_ax = None
        self.vg_chart = None
        # self.bar_function()
        # self.dis_function()

        self.create_tex()
        self.create_offer()
        self.bar_function()
        self.random_pick()
        self.move()
        self.max_value()

    def create_tex(self):
        prob_N = MathTex('P(a)=\\frac{1}{N}\sum_{b=0}^{N-a-1}\\frac{a}{a+b}').to_edge(2 * LEFT + 2 * UP)
        self.play(Write(prob_N))

        a_var = Variable(1, 'a', color=RED, var_type=Integer)
        p_var = Variable(0, 'P(a)', num_decimal_places=3)
        p_var.add_updater(lambda v: v.tracker.set_value(self.func1(a_var.tracker.get_value(), 50)))

        vg_var = VGroup(a_var, p_var)
        vg_var.arrange(DOWN, buff=0.8).to_edge(RIGHT)

        self.vg_tex = VGroup(prob_N, vg_var)

    def create_offer(self):
        line_ob = Line(start=3 * LEFT, end=2.5 * LEFT,
                       stroke_width=20, color=BLUE).to_edge(LEFT)
        svg_offer = SVGMobject('svg_icon/house_price.svg', fill_color=WHITE).scale(0.4)
        svg_people = SVGMobject('svg_icon/people.svg', fill_color=WHITE).scale(0.6).next_to(svg_offer, DOWN)
        vg_list = VGroup(svg_people, svg_offer).next_to(line_ob, UP)
        vg_list.add_updater(lambda m: m.next_to(line_ob, UP))
        vg_ob = VGroup(line_ob, vg_list)

        line_select = Line(start=10 * LEFT, end=3 * RIGHT,
                           stroke_width=20, color=BLUE).to_edge(RIGHT)
        svg_offer = SVGMobject('svg_icon/house_price.svg', fill_color=WHITE).scale(0.4)
        svg_people = SVGMobject('svg_icon/people.svg', fill_color=WHITE).scale(0.6).next_to(svg_offer, DOWN)
        vg_list2 = VGroup(svg_people, svg_offer).next_to(line_select, UP)
        vg_list2.add_updater(lambda m: m.next_to(line_select, UP))

        vg_select = VGroup(line_select, vg_list2)

        vg_line = VGroup(vg_ob, vg_select).scale(0.5).to_edge(2 * UP + RIGHT)

        self.play(FadeIn(vg_line))
        self.vg_line = vg_line

    @staticmethod
    def func1(r, n):
        s = 0
        res = int(n - r - 1)
        for m in range(res):
            s += r / (m + r)
        return s / n

    @staticmethod
    def func2(x):
        return -x * np.log(x)

    def bar_function(self):
        ax = CommonFunc.add_axes(x_range=[1, 50], y_range=[0, 1, 2], x_length=8, y_length=2,
                                 axis_config={"include_tip": False, "include_numbers": False}).scale(1.3).shift(
            DOWN).to_edge(LEFT)
        dashed_line = DashedLine(start=ax.c2p(0, 0.01), end=ax.c2p(50, 0.01), color=MAROON)
        line_label = MathTex('\\frac{1}{N}=\\frac{1}{100}', color=MAROON).scale(0.5).next_to(dashed_line, RIGHT)

        self.vg_ax = VGroup(ax, dashed_line, line_label)

        x_range = list(range(1, 50))
        values = [round(self.func1(i, 50), 3) for i in x_range]
        chart = BarChart(
            values=values,
            bar_names=None,
            y_range=[0, 1, 2],
            y_length=2,
            x_length=8,
            x_axis_config={"font_size": 12},
        ).scale(1.3).shift(DOWN).to_edge(LEFT)

        self.vg_chart = chart

    def random_pick(self):
        self.play(FadeIn(self.vg_ax[0], target_position=self.vg_tex[0]))

        vector_pick = Vector(UP).next_to(self.vg_line[0][0].get_left(), DOWN)
        self.play(DrawBorderThenFill(vector_pick))

        self.play(vector_pick.animate.next_to(self.vg_line[1][0].get_right(), DOWN), run_time=2)

        self.play(FadeTransform(vector_pick, self.vg_ax[1:]))
        self.wait(1)

    def move(self):
        vg_line = self.vg_line
        vg_tex = self.vg_tex
        vg_ax = self.vg_ax
        vg_chart = self.vg_chart

        self.play(vg_line[0][0].animate.set_color(RED))

        self.play(FadeIn(vg_tex[1][0], target_position=vg_line[0][0]))
        self.play(FadeTransform(vg_tex[0].copy(), vg_tex[-1][1]))

        right_gap = (vg_line[1][0].get_right() - vg_line[0][0].get_right()) * 19 / 20
        # a从1增加到100
        self.play(vg_line[0][0].animate.put_start_and_end_on(start=vg_line[0][0].get_left(),
                                                             end=vg_line[0][0].get_right() + right_gap),
                  vg_line[1][0].animate.put_start_and_end_on(start=vg_line[1][0].get_left() + right_gap,
                                                             end=vg_line[1][0].get_right()),
                  self.vg_tex[-1][0].tracker.animate.set_value(49),
                  Create(vg_chart[0]),
                  run_time=20)

        self.wait(1)

    def max_value(self):
        vg_chart = self.vg_chart
        vg_ax = self.vg_ax

        x_range = list(range(1, 50))
        values = [round(self.func1(i, 50), 3) for i in x_range]
        m = values.index(max(values))

        self.move_camera(frame_center=vg_chart[0][m], zoom=3)

        point = Dot(vg_ax[0].coords_to_point(x_range[m] + 0.5, values[m]), color=YELLOW).scale(0.3)
        lines = vg_ax[0].get_lines_to_point(point.get_center(), color=YELLOW)
        point_label = MathTex('({},{})'.format(x_range[m] + 1, values[m])).scale(0.3).next_to(point, UP)

        self.play(FadeIn(point, target_position=vg_chart[0][m]))
        self.play(GrowFromPoint(lines, point=point.get_center()))
        self.play(Write(point_label))

        self.wait(2)


class OSTCalculus(ThreeDScene):
    def construct(self):
        self.vg_tex = None
        self.vg_line = None
        self.vg_ax = None
        self.vg_chart = None
        self.vg_res = None
        self.chart_res = None

        self.create_chart_tex()
        self.trans_far()
        self.trans_calculus()
        self.calculus()
        self.calculus_anly()

    @staticmethod
    def func1(r, n):
        s = 0
        res = int(n - r - 1)
        for m in range(res):
            s += r / (m + r)
        return s / n

    @staticmethod
    def func2(x):
        return -x * np.log(x)

    def create_chart_tex(self):
        prob_N = MathTex('P(a)=', '\\frac{1}{N}', '\sum_{b=0}^{N-a-1}', '\\frac{a}{a+b}').scale(0.8).to_edge(UP)

        self.vg_tex = VGroup(prob_N)

        self.add(prob_N)

        ax = CommonFunc.add_axes(x_range=[1, 50], y_range=[0, 1, 2], x_length=8, y_length=2,
                                 axis_config={"include_tip": False, "include_numbers": False}).scale(1.5)
        self.vg_ax = VGroup(ax)

        self.add(ax)

        x_range = list(range(1, 50))
        values = [round(self.func1(i, 50), 3) for i in x_range]
        chart = BarChart(
            values=values,
            bar_names=None,
            y_range=[0, 1, 2],
            y_length=2,
            x_length=8,
            x_axis_config={"font_size": 12},
        ).scale(1.5)

        self.vg_chart = chart

        self.add(chart[0])

    def trans_far(self):

        n = Variable(var=50, label=MathTex('N'), var_type=Integer).next_to(self.vg_ax, DOWN)

        self.play(FadeIn(n, target_position=self.vg_chart))

        self.wait(1)

        for t in [60, 70, 80, 90, 100, 130, 150, 200, 250, 300]:
            x_range = list(range(1, t))
            values = [round(self.func1(i, t), 3) for i in x_range]
            new_chart = BarChart(
                values=values,
                bar_names=None,
                y_range=[0, 1, 2],
                y_length=2,
                x_length=8,
                x_axis_config={"font_size": 12},
            ).scale(1.5)
            self.play(Transform(self.vg_chart[0], new_chart[0]),
                      n.tracker.animate.set_value(t))
        self.wait(1)

        self.play(FadeOut(n))

    def trans_calculus(self):
        prob_N = self.vg_tex[0]
        self.play(prob_N.animate.to_edge(LEFT + 2 * UP))

        # 比例比绝对值更有意义
        r = MathTex('r=\\frac{a}{N}')
        t = MathTex('t=\\frac{b}{N}')

        vg_var = VGroup(r, t).arrange_submobjects(DOWN, buff=0.4).scale(0.5).next_to(prob_N, 3 * RIGHT)

        self.play(Write(vg_var))

        self.wait(2)

        # 可以写出积分式
        prob_C = MathTex('P(r)=', '\int_{0}^{1-r-\\frac{1}{n}}', '\\frac{r}{r+t} dt', color=YELLOW).scale(0.8).next_to(
            vg_var, 3 * RIGHT)
        line = Arrow(start=prob_N.get_right(), end=prob_C.get_left(),
                     max_stroke_width_to_length_ratio=2,
                     color=MAROON).scale(1.1)
        self.play(GrowArrow(line))
        self.wait(1)

        self.play(Indicate(prob_N[0]))
        self.play(FadeIn(prob_C[0], target_position=prob_N[0]))
        self.wait(1)
        self.play(Indicate(prob_N[2]))
        self.play(FadeIn(prob_C[1], target_position=prob_N[2]))
        self.wait(1)
        self.play(Indicate(prob_N[1]),
                  Indicate(prob_N[-1]))
        self.play(FadeTransform(VGroup(prob_N[1], prob_N[-1]).copy(), prob_C[-1]))

        # 最后的结果,简单的定积分
        self.play(Circumscribe(prob_C[1:], color=GREEN))
        self.wait(1)
        prob_res = MathTex('rln(1-\\frac{1}{n})', '-r\ln(r)', color=YELLOW).scale(0.8).next_to(prob_C[0], 0.5 * RIGHT)

        self.play(ReplacementTransform(prob_C[1:], prob_res))
        self.wait(2)

        self.play(prob_res[0].animate.set_color(GRAY))
        self.wait(1)

        self.play(FadeOut(prob_res[0]),
                  prob_res[-1].animate.move_to(prob_res[0].get_center()))
        self.wait(1)

        self.play(FadeOut(vg_var),
                  FadeOut(line),
                  FadeOut(self.vg_tex[0][0]),
                  FadeOut(prob_C[0]))

        self.vg_res = prob_res[-1]

    def calculus(self):
        prob_N = self.vg_tex[0][1:]
        origin_chart = VGroup(self.vg_chart[0], self.vg_ax)
        self.play(origin_chart.animate.scale(0.8).to_edge(LEFT).shift(2 * UP))

        ax2_calculus = CommonFunc.add_axes(x_range=[0, 1], y_range=[0, 1, 2], x_length=8, y_length=2,
                                           axis_config={"include_tip": False, "include_numbers": False}).scale(
            1.2).next_to(origin_chart, DOWN)

        self.play(Create(ax2_calculus))
        self.play(self.vg_res.animate.next_to(ax2_calculus, 2 * RIGHT))
        self.play(prob_N.animate.next_to(self.vg_ax, RIGHT))

        graph = ax2_calculus.plot(lambda x: self.func2(x), x_range=[0.0001, 1], use_smoothing=True, color=YELLOW)

        self.play(Write(graph))
        self.wait(2)

        self.chart_res = VGroup(ax2_calculus, graph)

    def calculus_anly(self):
        chart_res = self.chart_res

        gap = self.vg_ax[0].get_center() - chart_res[0].get_center()

        self.play(chart_res[-1].copy().animate.shift(gap))

        self.wait(1)

        self.move_camera(frame_center=chart_res, zoom=1.3)

        x_start = ValueTracker(0.01)

        point_start = chart_res[0].c2p(x_start.get_value(), self.func2(x_start.get_value()))
        moving_dot = Dot(point_start, color=GREEN).scale(0.5)

        tangent_line = chart_res[0].get_secant_slope_group(x=x_start.get_value(),
                                                           graph=chart_res[1],
                                                           dx=0.001,
                                                           secant_line_length=1.3,
                                                           secant_line_color=RED_D)

        tangent_line.add_updater(lambda m: m.become(chart_res[0].get_secant_slope_group(x=chart_res[0].p2c(moving_dot.get_center())[0],
                                                           graph=chart_res[1],
                                                           dx=0.001,
                                                           secant_line_length=1.3,
                                                           secant_line_color=RED_D)))
        derivative_value = Variable(0,
                                    label=MathTex('P^{\prime}'),
                                    var_type=DecimalNumber,
                                    ).scale(0.5).next_to(moving_dot, 1.5*UP)
        derivative_value.add_updater(lambda m: m.next_to(moving_dot, 1.5*UP))

        derivative_value.add_updater(lambda m: m.tracker.set_value(chart_res[0].p2c(moving_dot.get_center())[0]))

        self.add(moving_dot, tangent_line, derivative_value)

        self.play(MoveAlongPath(moving_dot, chart_res[-1]),
                  # UpdateFromFunc(tangent_line,
                  #                lambda line: line.become(chart_res[0].get_secant_slope_group(x=moving_dot.get_center()[0],
                  #                                          graph=chart_res[1],
                  #                                          dx=0.001,
                  #                                          secant_line_length=0.7,
                  #                                          secant_line_color=RED_D))),
                  run_time = 10)

        self.wait(2)

class thanks_end(Scene):
    def construct(self):
        svg_image = SVGMobject('svg_icon/bird.svg', fill_color=MAROON).scale(2.5)

        text = Text('迷路的小画家', font='SIL-Hei-Med-Jian').scale(1.5).next_to(svg_image, 2.5 * DOWN)

        self.play(SpinInFromNothing(VGroup(svg_image, text)))
