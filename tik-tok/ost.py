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
        arrow = Arrow(start=offer_group[0].get_left()+1.5*DOWN,
                      end=offer_group[-1].get_right()+1.5*DOWN,
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
        #self.anly()

    def offer_create(self):
        svg_offer = SVGMobject('svg_icon/house_price.svg', fill_color=WHITE).scale(0.5)
        svg_people = SVGMobject('svg_icon/people.svg', fill_color=BLUE).scale(0.7).next_to(svg_offer, DOWN)
        vg_list = VGroup(svg_people, svg_offer)
        offer_group = VGroup(*[vg_list.copy() for _ in range(10)])
        offer_group.arrange_submobjects(RIGHT, buff=0.45).scale(0.8)
        self.add(offer_group)
        self.offer_group = offer_group

        # brace = Brace(offer_group, direction=UP, color=YELLOW)
        # brace_text = MathTex('N').scale(0.9).next_to(brace, UP)
        #
        # self.add(brace, brace_text)
        # arrow = Arrow(start=offer_group[0].get_left() + 1.5 * DOWN,
        #               end=offer_group[-1].get_right() + 1.5 * DOWN,
        #               color=MAROON, max_stroke_width_to_length_ratio=50)
        # self.add(arrow)
        #
        # self.wait(1)
        # self.play(FadeOut(arrow))

    def group(self):
        offer_group = self.offer_group


        line_ob = Line(start=offer_group[0].get_left(),
                       end=offer_group[3].get_right(),
                       stroke_width=20, color=RED)
        svg_offer = SVGMobject('svg_icon/house_price.svg', fill_color=WHITE).scale(0.4)
        svg_people = SVGMobject('svg_icon/people.svg', fill_color=WHITE).scale(0.6).next_to(svg_offer, DOWN)
        vg_list = VGroup(svg_people, svg_offer).next_to(line_ob, UP)

        vg_ob = VGroup(line_ob, vg_list)

        self.play(ReplacementTransform(offer_group[:4], vg_ob))

        self.wait(1)

        line_select = Line(start=offer_group[4].get_left(),
                           end=offer_group[-1].get_right(),
                           stroke_width=20, color=BLUE)
        svg_offer = SVGMobject('svg_icon/house_price.svg', fill_color=WHITE).scale(0.4)
        svg_people = SVGMobject('svg_icon/people.svg', fill_color=WHITE).scale(0.6).next_to(svg_offer, DOWN)
        vg_list2 = VGroup(svg_people, svg_offer).next_to(line_select, UP)

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

        self.play(vector_ob.animate.next_to(vg_line[0][0].get_right(), DOWN), run_time=2)

        ## 依次选取，遇见一个比之前的更好，就选它！

        vector_pick = Vector(UP).next_to(vg_line[1][0].get_left(), DOWN)
        self.play(DrawBorderThenFill(vector_pick))
        self.wait(1)

        self.play(vector_pick.animate.next_to(vg_line[1][0].get_right()/2, DOWN), run_time=2)

    def anly(self):
        vg_line = self.vg_line

        brace_ob = Brace(vg_line[0][0], direction=DOWN, color=YELLOW)
        brace_text_ob = MathTex('r').next_to(brace_ob, DOWN)
        vg_brace_ob = VGroup(brace_ob, brace_text_ob)
        self.play(Write(vg_brace_ob))






class OST(Scene):
    def construct(self):
        l_offer = None
        self.bar_function()
        #self.dis_function()
        pass

    @staticmethod
    def func1(r, n):
        s = 0
        res = int(n-r)
        for m in range(res):
            s += r/(m+r)
        return s/n

    @staticmethod
    def func2(x):
        return -x*np.log(x)

    def bar_function(self):
        x_range = list(range(1,100))
        values = [round(self.func1(i,100),3) for i in x_range]
        inital_chart = BarChart(
                values=values,
                bar_names=None,
                y_range=[0, 1, 2],
                y_length=4,
                x_length=8,
                x_axis_config={"font_size": 12},
            ).scale(1.3).to_edge(LEFT)
        #c_bar_lbls = inital_chart.get_bar_labels(font_size=15)
        self.play(Create(inital_chart))

        self.wait(3)

    def dis_function(self):
        ax = CommonFunc.add_axes(x_range=[0, 100], y_range=[0, 1], x_length=8, y_length=8,
                                 axis_config={"include_tip": True, "include_numbers": False}).scale(1.2)
        graph = ax.plot(lambda x: self.func1(x, 100), x_range=[1, 99], use_smoothing=True)
        graph_label = ax.get_graph_label(graph=graph,
                                         label=MathTex('\mu'),
                                         direction=DL)

        ax_vg = VGroup(ax, graph, graph_label)

        self.play(Create(ax_vg))

        self.wait(3)



class thanks_end(Scene):
    def construct(self):
        svg_image = SVGMobject('svg_icon/bird.svg', fill_color=MAROON).scale(2.5)

        text = Text('迷路的小画家', font='SIL-Hei-Med-Jian').scale(1.5).next_to(svg_image, 2.5 * DOWN)

        self.play(SpinInFromNothing(VGroup(svg_image, text)))

