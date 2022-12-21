from manim import *
import numpy as np
import sys

sys.path.append('..')

from CS_learning.common_func import CommonFunc


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

        self.play(group.animate.scale(0.6).shift(UP))

        phanton_group = VGroup(*[CommonFunc.add_function(lambda x: 0.3 * np.sin(5 * x - np.pi * 0.5 * dx),
                                                         x_range=(-3, 3)).scale(0.6).shift(UP) for dx in range(41)])

        text_cn = Text('假装学习', font='HiraginoSansGB-W3').scale(0.8).next_to(group, DOWN)
        text_en = Text('Pretending to Learn').scale(0.8).next_to(text_cn, DOWN)

        self.play(FadeIn(text_cn), Create(text_en))

        self.play(Succession(*[Transform(phanton, phanton_group[i]) for i in range(len(phanton_group))], run_time=5))


class Title(Scene):
    def construct(self):
        svg_object = SVGMobject('svg_icon/book.svg', fill_color=BLUE)
        svg_group = VGroup(*[svg_object.copy() for _ in range(10)]).scale(0.4)
        svg_group.arrange_submobjects(RIGHT, buff=0.2).shift(2 * UP)

        self.play(Create(svg_group))
        self.wait(5)

        self.play(Indicate(svg_group[0], run_time=2))

        text = Text('枚举算法    Enumeration Algorithm ').next_to(svg_group, DOWN * 4)

        self.play(GrowFromPoint(text, svg_group[0].get_center(), run_time=2))
        self.wait(3)

        subtext = Text('-- 列举出问题所有可能的解，再进行筛选').scale(0.5).next_to(text, 1.5*DOWN)
        self.play(Write(subtext))


class enumerate_example(Scene):
    def construct(self):
        l_n = list(range(1, 101))
        circle_number = CommonFunc.add_shape_object(l_n).scale(0.55)
        self.play(FadeIn(circle_number, lag_ratio=0.5))

        def if_prime(n):
            for j in range(2, int(np.sqrt(n)) + 1):
                if n % j == 0:
                    return False
            return True

        circles = circle_number[0]
        numbers = circle_number[1]
        r = circles[0].get_radius()
        for i in range(len(numbers)):
            if if_prime(numbers[i].get_value()):
                circles[i].become(Circle(radius=r, color=RED).scale(0.55), match_center=True)
                self.wait(1)

        self.play(circle_number.animate.shift(3.5 * LEFT))

        text0_cn = Text('枚举算法的要素', font='SIL-Hei-Med-Jian').scale(0.8).next_to(circle_number, 7 * RIGHT + 0.5 * UP)
        text0_en = Text('Elements of enumeration algorithm').scale(0.5).next_to(text0_cn, DOWN)
        self.play(Write(text0_cn), Write(text0_en))
        self.wait(5)

        text1_cn = Text('1. 减少枚举空间').scale(0.5).next_to(text0_en, 3 * DOWN)
        text1_en = Text('reduce enumeration scope').scale(0.4).next_to(text1_cn, DOWN)

        text2_cn = Text('2. 优化筛选方法').scale(0.5).next_to(text1_en, 3 * DOWN)
        text2_en = Text(' optimize filtering methods').scale(0.4).next_to(text2_cn, DOWN)

        text3_cn = Text('3. 选择合适枚举顺序').scale(0.5).next_to(text2_en, 3 * DOWN)
        text3_en = Text('choose an appropriate enumeration order').scale(0.4).next_to(text3_cn, DOWN)

        self.play(Write(text1_cn), Write(text1_en),
                  Write(text2_cn), Write(text2_en),
                  Write(text3_cn), Write(text3_en))


class source_code(Scene):
    def construct(self):
        title = Text('例子:', font='SIL-Hei-Med-Jian').scale(0.8).move_to(np.array([-5, 3.5, 0]))
        self.add(title)
        subtitle = Text('一个数组中的数互不相同，其中和为0的数对有多少对').scale(0.6).next_to(title, RIGHT)
        self.add(subtitle)
        topic = Paragraph('\t Given an array of numbers where each number is unique,',
                          '\t find the number of pairs of numbers in the array that sum to 0.').set_color(
            MAROON).scale(0.5).next_to(subtitle, DOWN)
        self.play(FadeIn(topic))

        l_n = list(range(-5, 5))
        integers_1 = CommonFunc.get_Integer(l_n).scale(0.8).shift(3 * RIGHT)
        integers_2 = CommonFunc.get_Integer(l_n, color=RED).scale(0.8).next_to(integers_1, DOWN * 2)

        self.add(integers_1)
        self.wait(2)
        self.play(FadeIn(integers_2))

        # 追踪数据点（外层循环）
        pointer_1, tracker_1, label_1 = CommonFunc.pointer_tracker(integers_1)
        self.play(FadeIn(label_1), Write(pointer_1))

        # 追踪数据点（内层循环）
        y_2 = integers_2[0].get_center()[1]
        pointer_2, tracker_2, label_2 = CommonFunc.pointer_tracker(integers_2, label_name='y', y=y_2, direction=UP,
                                                                   position=DOWN)
        self.play(FadeIn(label_2), Write(pointer_2))

        code = CommonFunc.add_code('enumerate/code1.py', 'python').next_to(integers_1, LEFT * 3.5)
        self.play(Create(code))

        var = CommonFunc.variable_tracker(label=Tex('$\\text{res}$'), color=GREEN).next_to(code, UP)
        self.play(Create(var))

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

        time_consume = Text('时间复杂度（Time Complexity)').scale(0.6).next_to(code, DOWN)
        time_complex = Tex('$n^2$').scale(0.8).next_to(time_consume, RIGHT)
        self.play(Write(time_consume), Write(time_complex))
        self.wait()


class fast_code(Scene):
    def construct(self):
        text = Text('重复计算(Double counting)')
        self.play(text.animate.scale(0.8).shift(3 * UP))
        tex = Tex('$a+b=b+a$').next_to(text, DOWN)
        self.play(Write(tex))

        self.wait(3)

        l_n = list(range(-5, 5))
        integers_1 = CommonFunc.get_Integer(l_n).scale(1.2)
        integers_2 = CommonFunc.get_Integer(l_n, color=RED).scale(1.2).next_to(integers_1, DOWN * 5)
        self.add(integers_1, integers_2)

        n = len(integers_1)
        wait_delete = VGroup()
        for i in range(n):
            line = CommonFunc.add_line(integers_1[0], integers_2[i])
            dashline = CommonFunc.add_dashed_line(integers_1[i], integers_2[0])

            self.play(Create(line))
            coor1 = Text('({},{})'.format(integers_1[0].get_value(),
                                          integers_2[i].get_value())).scale(0.4).next_to(integers_2[i], DOWN)
            self.play(Write(coor1))
            self.play(Create(dashline))
            coor2 = Text('({},{})'.format(integers_1[i].get_value(),
                                          integers_2[0].get_value())).scale(0.4).next_to(integers_1[i], UP)
            self.play(Write(coor2))
            self.play(Uncreate(line), Uncreate(dashline))
            wait_delete.add(coor1, coor2)

        self.play(FadeOut(wait_delete))

        zero_situation = Text('({},{})'.format(0, 0)).scale(0.6).next_to(integers_1[5], UP)
        self.play(Write(zero_situation))

        zero_explain = Tex('$0+0=0$').scale(0.6).next_to(zero_situation, 2 * UP + 4 * RIGHT)
        arrow = CommonFunc.add_arrow(zero_explain, zero_situation, buff=0.1)
        self.play(GrowFromCenter(zero_explain), Write(arrow))
        self.play(FocusOn(zero_situation, run_time=5))


class handshake(Scene):
    def construct(self):
        title = Text('减少重复计算(reduce double counting)').scale(0.8).shift(3 * UP)
        self.add(title)
        l_n = list(range(-5, 5))
        integers_1 = CommonFunc.get_Integer(l_n).scale(0.8).shift(3 * RIGHT)
        integers_2 = CommonFunc.get_Integer(l_n, color=RED).scale(0.8).next_to(integers_1, DOWN * 2)

        self.play(Create(integers_1), Create(integers_2))

        # 追踪数据点（外层循环）
        pointer_1, tracker_1, label_1 = CommonFunc.pointer_tracker(integers_1)
        self.play(FadeIn(label_1), Write(pointer_1))

        # 追踪数据点（内层循环）
        y_2 = integers_2[0].get_center()[1]
        pointer_2, tracker_2, label_2 = CommonFunc.pointer_tracker(integers_2, label_name='y', y=y_2, direction=UP,
                                                                   position=DOWN)
        self.play(FadeIn(label_2), Write(pointer_2))

        code = CommonFunc.add_code('enumerate/code2.py', 'python').next_to(integers_1, LEFT * 3.5)
        self.play(Create(code))

        var = CommonFunc.variable_tracker(label=Tex('$\\frac{\\text{res}}{2}$'), color=GREEN).next_to(code, UP)
        self.add(var)

        sum_result = 0
        # 第一个循环
        for i in range(len(integers_1)):
            # 对应的代码进行闪烁
            self.play(tracker_1.animate.set_value(integers_1[i].get_center()[0]))
            self.play(Indicate(code.code[0]))
            # 第二个循环
            for j in range(i + 1, len(integers_2)):
                # 对应的代码进行闪烁
                self.play(tracker_2.animate.set_value(integers_2[j].get_center()[0]))
                self.play(Indicate(code.code[1]))
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

        self.play(FadeOut(var))
        final_var = Tex('$\\text{res}=8$').next_to(code, UP)
        self.play(FadeIn(final_var))

        time_consume = Text('时间复杂度（Time Complexity)').scale(0.6).next_to(code, DOWN)
        time_complex = Tex('$\\frac{n^2}{2}$').scale(0.8).next_to(time_consume, RIGHT)
        self.play(Write(time_consume), Write(time_complex))
        self.wait()


class faster_hash(Scene):
    def construct(self):
        title = Text('空间换时间（Trade space for time）').scale(0.8).shift(3 * UP)
        self.add(title)
        l_n = list(range(-5, 5))
        integers_1 = CommonFunc.get_Integer(l_n).scale(0.8).shift(3 * RIGHT)
        self.add(integers_1)

        source_table = CommonFunc.add_table([[Tex('$\\text{key}$')],
                                             [Tex('$\\text{value}$')]]).next_to(integers_1, 5 * DOWN).scale(0.6)

        pointer_1, tracker_1, label_1 = CommonFunc.pointer_tracker(integers_1)
        self.play(Create(pointer_1), FadeIn(label_1))

        code = CommonFunc.add_code('enumerate/code3.py', 'python').next_to(integers_1, LEFT * 0.2)
        self.play(Create(code))

        var = CommonFunc.variable_tracker(label=Tex('$\\frac{\\text{res}}{2}$'), color=GREEN).next_to(code, UP)
        self.add(var)

        self.play(Indicate(code.code[0]))
        self.play(Create(source_table))
        self.play(FadeOut(source_table))
        hash_map = {}
        sum_res = 0
        table_group = VGroup()
        for i in range(len(integers_1)):
            # 对应的代码进行闪烁
            self.play(tracker_1.animate.set_value(integers_1[i].get_center()[0]))
            self.play(Indicate(code.code[1]))

            self.play(Indicate(code.code[2]))
            if hash_map.get(0 - integers_1[i].get_value()):
                self.play(Indicate(code.code[3]))
                sum_res += 1
                self.play(var.tracker.animate.set_value(sum_res))
            else:
                self.play(Indicate(code.code[5]))
                table = CommonFunc.add_table([[Tex('${}$'.format(integers_1[i].get_value()))],
                                              [Text('True').scale(0.7)]]).next_to(integers_1, 4 * DOWN).scale(0.5)
                self.play(FadeIn(table))
                hash_map[integers_1[i].get_value()] = True
                table_group.add(table)
                self.play(FadeOut(table))

        table_group.arrange_submobjects(RIGHT, buff=0.05).next_to(integers_1, 7 * DOWN)
        self.play(Create(table_group))

        time_consume = Text('时间复杂度（Time Complexity)').scale(0.5).next_to(code, DOWN)
        time_complex = Tex('$n$').scale(0.8).next_to(time_consume, DOWN)
        self.play(Write(time_consume), Write(time_complex))
        self.wait()


class time_compare(Scene):
    def construct(self):
        ax = CommonFunc.add_axes(x_range=[1, 10], y_range=[0, 150, 10], x_length=8, y_length=6)

        def func(x):
            return x ** 2

        def func_linear(x):
            return x

        def func_sum(x):
            return x ** 2 / 2

        self.play(Create(ax))
        self.wait()

        l_func = [func, func_sum, func_linear]
        texs = ["$n^2$", "$\\frac{n^2}{2}$", '$n$']
        colors = [YELLOW, GOLD, MAROON]

        for i in range(len(l_func)):
            graph = ax.plot(l_func[i], x_range=[1, 10], color=colors[i], use_smoothing=True)
            graph_label = ax.get_graph_label(graph=graph, label=Tex(texs[i]))
            self.play(Create(graph), Write(graph_label))
            self.wait()


class screen(Scene):
    def construct(self):
        svg = SVGMobject('images/bird.svg').scale(2)

        text = Text('像小鸟一样努力').next_to(svg, DOWN)

        self.play(SpinInFromNothing(svg))
