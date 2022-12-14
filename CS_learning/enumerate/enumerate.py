from manim import *
import numpy as np
import sys

sys.path.append('..')

from CS_learning.common_func import CommonFunc


class Title(Scene):
    def construct(self):
        text = Text('Enumeration Algorithm').scale(1.5)
        self.play(Write(text))
        self.wait(2)


class source_code(Scene):
    def construct(self):
        title = Text('An Example').scale(0.8).move_to(np.array([-5, 3.5, 0]))
        self.add(title)

        l_n = list(range(-5, 5))
        integers_1 = CommonFunc.get_Integer(l_n).scale(0.8).shift(3 * RIGHT)
        integers_2 = CommonFunc.get_Integer(l_n, color=RED).scale(0.8).next_to(integers_1, DOWN * 2)

        self.add(integers_1)
        self.wait(2)
        self.play(FadeIn(integers_2))

        # 追踪数据点（外层循环）
        pointer_1, tracker_1, label_1 = CommonFunc.pointer_tracker(integers_1)
        self.add(pointer_1, label_1)

        # 追踪数据点（内层循环）
        y_2 = integers_2[0].get_center()[1]
        pointer_2, tracker_2, label_2 = CommonFunc.pointer_tracker(integers_2, label_name='y', y=y_2, direction=UP,
                                                                   position=DOWN)
        self.add(pointer_2, label_2)

        var = CommonFunc.variable_tracker(label=Tex('$\\text{res}$'), color=GREEN).next_to(integers_1, 9 * UP)
        self.add(var)

        code = CommonFunc.add_code('enumerate/code1.py', 'python').next_to(integers_1, LEFT * 3.5)
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
            for j in range(i+1, len(integers_2)):
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


class FastPower(Scene):
    def construct(self):
        pass


class screen(Scene):
    def construct(self):
        svg = SVGMobject('images/bird.svg').scale(2)

        text = Text('像小鸟一样努力').next_to(svg, DOWN)

        self.play(SpinInFromNothing(svg))
