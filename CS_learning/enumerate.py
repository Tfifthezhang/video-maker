from manim import *
import numpy as np

import networkx as nx


def get_integer_shape(l_n, color=BLUE):
    n_circles = len(l_n)

    circles = VGroup(*[Circle(radius=0.5,
                              color=color
                              )
                       for _ in range(n_circles)
                       ]
                     )
    circles.arrange_submobjects(RIGHT, buff=0.3)

    texs = VGroup()
    for i in range(n_circles):
        integer = Integer(number=l_n[i])
        integer.move_to(circles[i].get_center())
        texs.add(integer)

    circle_texs = VGroup(circles, texs)

    return circle_texs


def get_Integer(l_n, color=BLUE):
    integers = VGroup(*[Integer(number=i,
                                color=color,
                                ) for i in l_n
                        ]
                      )
    integers.arrange_submobjects(RIGHT, buff=0.5)
    return integers

def pointer_tracker(vg_object,label_name='x',y = 0):
    # 默认在y=0的平面上
    # 一个在上方的向下的箭头
    pointer = Vector(DOWN).next_to(vg_object[0].get_center(), UP)
    # 标签在pointer上方
    label = Text(label_name).add_updater(lambda m: m.next_to(pointer, UP))
    # 初始化valuetracker
    tracker = ValueTracker(0)
    # 将 pointer 绑定在 valuetracker上，默认
    pointer.add_updater(lambda m: m.next_to(np.array([tracker.get_value(), y, 0]), UP))

    return pointer, tracker, label


class Title(Scene):
    def construct(self):
        text = Text('Enumeration Algorithm').scale(1.5)
        self.play(Write(text))
        self.wait(2)


class number_in_circle(Scene):
    def construct(self):
        l_n = list(range(-5, 5))
        integers_1 = get_Integer(l_n)

        integers_2 = get_Integer(l_n, color=RED).next_to(integers_1, DOWN * 2)

        self.add(integers_1)
        self.wait(2)
        self.play(FadeIn(integers_2))

        # 追踪数据点（外层循环）
        pointer_1 = Vector(DOWN).next_to(integers_1[0].get_center(), UP)
        label_1 = Text('x').add_updater(lambda m: m.next_to(pointer_1, UP))
        #
        tracker_1 = ValueTracker(0)
        pointer_1.add_updater(lambda m: m.next_to(np.array([tracker_1.get_value(),0,0]), UP))
        #
        self.add(pointer_1, label_1)

        # 追踪数据点（内层循环）

        for i in range(len(integers_1)):
            self.play(tracker_1.animate.set_value(integers_1[i].get_center()[0]))


class DrawAndGate(Scene):
    def construct(self):
        pass


class generate2(Scene):
    def construct(self):
        pass


class FastPower(Scene):
    def construct(self):
        pass


class screen(Scene):
    def construct(self):
        svg = SVGMobject('images/bird.svg').scale(2)

        text = Text('像小鸟一样努力').next_to(svg, DOWN)

        self.play(SpinInFromNothing(svg))
