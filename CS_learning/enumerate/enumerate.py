from manim import *
import numpy as np
import math
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
        self.play(GrowFromPoint(phanton, start))
        self.wait()
        self.play(GrowArrow(q_average))
        self.play(GrowArrow(q))

        text = Text('pretending to learn').scale(0.8).next_to(phanton, UP)
        group = VGroup(phanton, e_minus, e_plus, q_average, q, text)

        self.play(Write(text))


class Title(Scene):
    def construct(self):
        text = Text('Enumeration Algorithm').scale(1.5)
        self.play(Write(text))
        self.wait(2)


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

        text0 = Text('Elements of enumeration algorithm:').scale(0.5).next_to(circle_number, RIGHT + 0.5 * UP)
        text1 = Text('1. Minimize the enumeration range').scale(0.4).next_to(text0, 3 * DOWN)
        text2 = Text('2. Optimize selection algorithm').scale(0.4).next_to(text1, 3 * DOWN)
        text3 = Text('3. choose an appropriate enumeration order').scale(0.4).next_to(text2, 3 * DOWN)

        self.play(Write(text0), Write(text1), Write(text2), Write(text3))

        self.wait(5)


class source_code(Scene):
    def construct(self):
        title = Text('An Example').scale(0.8).move_to(np.array([-5, 3.5, 0]))
        self.add(title)
        topic = Paragraph('\t Given an array of numbers where each number is unique,',
                          ' \t find the number of pairs of numbers in the array that sum to 0.').set_color(
            MAROON).scale(0.4).next_to(title, DOWN + 0.5 * RIGHT)
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

        var = CommonFunc.variable_tracker(label=Tex('$\\text{res}$'), color=GREEN).next_to(integers_1, 9 * UP)
        self.play(Create(var))

        code = CommonFunc.add_code('enumerate/code1.py', 'python').next_to(integers_1, LEFT * 3.5)
        self.play(Create(code))

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

        time_complex = Tex('Time Complexity:$n^2$').scale(0.6).next_to(code, DOWN)
        self.play(Write(time_complex))
        self.wait()


class fast_code(Scene):
    def construct(self):
        text = Text('Double counting')
        self.play(text.animate.scale(0.8).shift(3 * UP))

        tex = Tex('$a+b=b+a$').next_to(text, DOWN)
        self.play(Write(tex))

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

        zero_explain = Text('0 plus itself equals 0').scale(0.6).next_to(zero_situation, 2 * UP + 4 * RIGHT)
        arrow = CommonFunc.add_arrow(zero_explain, zero_situation, buff=0.1)
        self.play(GrowFromCenter(zero_explain), Write(arrow))
        self.play(FocusOn(zero_situation, run_time=5))


class handshake(Scene):
    def construct(self):
        l_n = list(range(-5, 5))
        integers_1 = CommonFunc.get_Integer(l_n).scale(0.8).shift(3 * RIGHT)
        integers_2 = CommonFunc.get_Integer(l_n, color=RED).scale(0.8).next_to(integers_1, DOWN * 2)

        self.add(integers_1, integers_2)

        # 追踪数据点（外层循环）
        pointer_1, tracker_1, label_1 = CommonFunc.pointer_tracker(integers_1)
        self.add(pointer_1, label_1)

        # 追踪数据点（内层循环）
        y_2 = integers_2[0].get_center()[1]
        pointer_2, tracker_2, label_2 = CommonFunc.pointer_tracker(integers_2, label_name='y', y=y_2, direction=UP,
                                                                   position=DOWN)
        self.add(pointer_2, label_2)

        var = CommonFunc.variable_tracker(label=Tex('$\\frac{\\text{res}}{2}$'), color=GREEN).next_to(integers_1,
                                                                                                      9 * UP)
        self.add(var)

        code = CommonFunc.add_code('enumerate/code2.py', 'python').next_to(integers_1, LEFT * 3.5)
        self.play(Create(code))

        text = 'Given an array of numbers where each number is unique,' \
               ' find the number of pairs of numbers in the array that sum to 0.'

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
        final_var = Tex('$\\text{res}=8$').next_to(integers_1, 9 * UP)
        self.play(FadeIn(final_var))

        time_complex = Tex('Time Complexity:$\\frac{n^2}{2}$', color=GREEN).scale(0.6).next_to(code, DOWN)
        self.play(Write(time_complex))
        self.wait()


class faster_hash(Scene):
    def construct(self):
        l_n = list(range(-5, 5))
        integers_1 = CommonFunc.get_Integer(l_n).scale(0.8).shift(3 * RIGHT)
        self.add(integers_1)

        var = CommonFunc.variable_tracker(label=Tex('$\\text{res}$'), color=GREEN).next_to(integers_1, 9 * UP)
        self.add(var)
        #
        source_table = CommonFunc.add_table([[Tex('$\\text{key}$')],
                                             [Tex('$\\text{value}$')]]).next_to(integers_1, 5 * DOWN).scale(0.6)

        pointer_1, tracker_1, label_1 = CommonFunc.pointer_tracker(integers_1)
        self.play(Create(pointer_1), FadeIn(label_1))

        code = CommonFunc.add_code('enumerate/code3.py', 'python').next_to(integers_1, LEFT * 0.5)
        self.play(Create(code))

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

        time_complex = Tex('Time Complexity:$n$', color=GREEN).scale(0.6).next_to(code, DOWN)
        self.play(Write(time_complex))

class FrameTimefunction(Scene):
    def construct(self):
        ax = Axes(x_range=[1, 10], y_range=[0, 150, 10],
                  x_length=8, y_length=6,
                  axis_config={"include_tip": True,
                               "include_numbers": True}
                  )

        # labels = ax.get_axis_labels(x_label="x", y_label="y")

        ax.move_to(np.array([-2, 0, 0]))

        def func(x):
            return x ** 2

        def func_linear(x):
            return x

        def func_log(x):
            return np.log(x) + 1

        def func_linear_log(x):
            return x * np.log(x)

        def func_n_Factorial(x):
            return np.power(x, 6)

        self.play(Create(ax))
        self.wait()

        l_func = [func_log, func_linear, func_linear_log, func, func_n_Factorial]
        texs = ["$\log n$", "$n$", "$n \log n$", "$n^2$", "$n!$"]
        colors = [TEAL, GREEN, YELLOW, GOLD, MAROON]

        for i in range(len(l_func)):
            graph = ax.plot(l_func[i], x_range=[1, 9], color=colors[i], use_smoothing=True)
            graph_label = ax.get_graph_label(graph=graph, label=Tex(texs[i]))
            self.play(Create(graph), Write(graph_label))
            self.wait()



class screen(Scene):
    def construct(self):
        svg = SVGMobject('images/bird.svg').scale(2)

        text = Text('像小鸟一样努力').next_to(svg, DOWN)

        self.play(SpinInFromNothing(svg))
