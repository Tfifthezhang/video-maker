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


def pointer_tracker(vg_object, label_name='x', y=0, direction=DOWN, position=UP):
    # 默认在y=0的平面上
    # 一个在上方的向下的箭头
    pointer = Vector(direction).next_to(vg_object[0].get_center(), position)
    # 标签在pointer上方
    label = Text(label_name).add_updater(lambda m: m.next_to(pointer, position))
    # 初始化valuetracker
    tracker = ValueTracker(vg_object[0].get_center()[0])
    # 将 pointer 绑定在 valuetracker上
    pointer.add_updater(lambda m: m.next_to(np.array([tracker.get_value(), y, 0]), position))

    return pointer, tracker, label


def variable_tracker(label='x', start=0, color=RED):
    var = Variable(start, Text(label), var_type=Integer)
    var.set_color(color)
    return var


def add_code(code_path, language):
    listing = Code(
        code_path,
        tab_width=4,
        background_stroke_width=1,
        font_size=25,
        background_stroke_color=WHITE,
        line_spacing=0.5,
        insert_line_no=True,
        style=Code.styles_list[15],
        background="window",
        language=language,
    )
    return listing


class Title(Scene):
    def construct(self):
        text = Text('Enumeration Algorithm').scale(1.5)
        self.play(Write(text))
        self.wait(2)


class number_in_circle(Scene):
    def construct(self):
        title = Text('An Example').scale(0.8).move_to(np.array([-5, 3.5, 0]))
        self.add(title)

        l_n = list(range(-5, 5))
        integers_1 = get_Integer(l_n).scale(0.8).shift(3 * RIGHT)
        integers_2 = get_Integer(l_n, color=RED).scale(0.8).next_to(integers_1, DOWN * 2)

        self.add(integers_1)
        self.wait(2)
        self.play(FadeIn(integers_2))

        # 追踪数据点（外层循环）
        pointer_1, tracker_1, label_1 = pointer_tracker(integers_1)
        self.add(pointer_1, label_1)

        # 追踪数据点（内层循环）
        y_2 = integers_2[0].get_center()[1]
        pointer_2, tracker_2, label_2 = pointer_tracker(integers_2, label_name='y', y=y_2, direction=UP, position=DOWN)
        self.add(pointer_2, label_2)

        var = variable_tracker(label='res', color=GREEN).next_to(integers_1, 9 * UP)
        self.add(var)

        code = add_code('enumerate/code1.py', 'python').next_to(integers_1, LEFT * 3.5)
        self.play(Create(code))

        text = 'Given an array of numbers where each number is unique,' \
               ' find the number of pairs of numbers in the array that sum to 0.'

        sum_result = 0
        # 第一个循环
        for i in range(len(integers_1)):
            # 对应的代码进行闪烁
            self.play(Indicate(code.code[0]))
            self.play(tracker_1.animate.set_value(integers_1[i].get_center()[0]))
            # 第二个循环
            for j in range(len(integers_2)):
                # 对应的代码进行闪烁
                self.play(Indicate(code.code[1]))
                self.play(tracker_2.animate.set_value(integers_2[j].get_center()[0]))
                if integers_1[i].get_value() + integers_2[j].get_value() == 0:
                    # 添加动画效果，表示符合条件
                    self.play(Indicate(code.code[2]))
                    self.play(ApplyWave(label_1), ApplyWave(label_2))
                    # 对记分板进行更新
                    sum_result += 1
                    # 对应的代码进行闪烁
                    self.play(Indicate(code.code[3]))
                    # variable进行跟踪
                    self.play(var.tracker.animate.set_value(sum_result))

        time_complex = Tex('时间复杂度(Time Complexity):$n^2$').next_to(code, DOWN)
        self.play(Write(time_complex))
        self.wait()


class fast_code(Scene):
    def construct(self):
        l_n = list(range(-5, 5))
        integers_1 = get_Integer(l_n).scale(0.8)
        integers_2 = get_Integer(l_n, color=RED).scale(0.8).next_to(integers_1, DOWN * 3)
        self.add(integers_1, integers_2)




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
