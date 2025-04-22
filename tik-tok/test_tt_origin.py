from manim import *
import numpy as np

import networkx as nx
from random import shuffle


class Title(Scene):
    pass


class BubbleSort_TT(Scene):
    def construct(self):
        self.before =None
        self.target = None
        self.history = None

        self.source_data()
        self.target_data()
        self.bubble_sort()
        self.time_complexity()

    def source_data(self):
        l_array = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        n_circles = len(l_array)

        circles = VGroup(*[Circle(radius=0.75,
                                  ).scale(1.2)
                           for _ in range(n_circles)
                           ]
                         )
        circles.arrange_submobjects(DOWN, buff=0.3)

        texs = VGroup()
        for i in range(n_circles):
            integ = Integer(number=l_array[i]).scale(1.5)
            integ.move_to(circles[i].get_center())
            texs.add(integ)

        circle_texs = VGroup(circles, texs)

        self.play(Create(circle_texs))

        self.wait(2)

        self.play(circle_texs.animate.to_edge(1.5*LEFT))
        self.wait(1)

        self.before = circle_texs

    def target_data(self):
        l_array = list(range(10))
        n_circles = len(l_array)

        circles = VGroup(*[Circle(radius=0.75, color=BLUE).scale(1.2)
                           for _ in range(n_circles)
                           ])
        circles.arrange_submobjects(DOWN, buff=0.3)

        texs = VGroup()
        for i in range(n_circles):
            integ = Integer(number=l_array[i]).scale(1.5)
            integ.move_to(circles[i].get_center())
            texs.add(integ)

        circle_texs = VGroup(circles, texs).to_edge(1.5*RIGHT)

        self.target = circle_texs

        self.play(FadeTransform(self.before.copy(), self.target))

    def bubble_sort(self):
        title = Text('冒泡排序Bubble Sort').scale(1.5).to_edge(1.5*UP)
        self.play(Create(title))
        self.wait(1)

        self.play(FadeOut(title))

        n_circles = 10
        circles = self.before[0]
        texs = self.before[1]

        sort_history = VGroup()
        for item in range(10):
            boxs = VGroup()
            for i in range(n_circles - 1 - item):
                framebox = SurroundingRectangle(circles[i:i + 2], buff=.1, color=YELLOW)
                boxs.add(framebox)
            if not boxs:
                break
            self.add(boxs[0])
            for i in range(len(boxs)):
                if texs[i].get_value() > texs[i + 1].get_value():
                    self.play(Swap(texs[i], texs[i + 1]))
                    temp = texs[i]
                    texs[i] = texs[i + 1]
                    texs[i + 1] = temp
                if i <= len(boxs) - 2:
                    self.play(ReplacementTransform(boxs[i], boxs[i + 1]))
            self.play(FadeOut(boxs[-1]))
            move_circles_texs = self.before.copy()
            sort_history.add(move_circles_texs)
            self.play(move_circles_texs.animate.scale(0.3).move_to(self.before.get_corner(RIGHT)).shift(0.8*(item+1)*RIGHT))
            #self.play(Rotate(move_circles_texs, angle=PI, about_point=self.before.center()))

        self.play(FadeTransform(sort_history[-1].copy(), self.target))
        self.wait(1)

        self.history = sort_history

    def time_complexity(self):
        self.play(FadeOut(self.before, self.target))

        self.play(self.history.animate.scale(1.5).move_to(ORIGIN))

        for i in range(len(self.history)):
            l_circle_tex = self.history[i]
            circles = l_circle_tex[0]
            l = len(circles)
            self.play(circles[0:l-1-i].animate.set_color(YELLOW_A))

        self.wait(1)

        self.play(self.history.animate.to_edge(LEFT))
        brace_out = Brace(self.history, direction=RIGHT, color=MAROON)
        text_out = Tex('$n=10$').scale(1.2).next_to(brace_out, RIGHT)
        brace_in = Brace(self.history, direction=DOWN, color=MAROON)
        text_in = Tex('$n-1=9$').scale(1.2).next_to(brace_in, DOWN)
        self.play(Write(brace_in), Write(text_in), Write(brace_out), Write(text_out))
        self.wait()

        self.play(VGroup(self.history,
                         brace_out,
                         text_out,
                         brace_in,
                         text_in).animate.shift(4*UP))
        self.wait(1)
        tc_text = Text('时间复杂度Time Complexity').scale(1.5).to_edge(3*DOWN)
        tc_tex = Tex('$\\frac{n(n-1)}{2}$').scale(3).next_to(tc_text, DOWN).shift(1.5*LEFT)

        self.play(FadeTransform(VGroup(text_out, text_in), tc_tex))
        self.wait(1)

        self.play(FadeTransform(self.history.copy(), tc_text))
        self.wait(2)

        tc_jianjin = Tex('$\sim O(n ^ 2)$').scale(2.5).next_to(tc_tex, RIGHT)
        self.play(FadeIn(tc_jianjin))
        self.wait(1)

class thanks_end(Scene):
    def construct(self):
        svg_image = SVGMobject('svg_icon/bird.svg', fill_color=MAROON).scale(2.5)

        text = Text('迷路的小画家', font='SIL-Hei-Med-Jian').scale(1.5).next_to(svg_image, 2.5 * DOWN)

        self.play(SpinInFromNothing(VGroup(svg_image, text)))

