from manim import *
import numpy as np

import networkx as nx
from random import shuffle


class Title(Scene):
    pass


class BubbleSort_TT(Scene):
    # CONFIG = {
    #     "frame_width":9,
    #     "frame_height":16
    # }
    def construct(self):
        # self.camera.frame_width = 1080  # 宽度
        # self.camera.frame_height = 1920 # 高度
        l_array = [9, 5, 8, 3, 0, 6, 1, 4, 2, 7]
        n_circles = len(l_array)

        circles = VGroup(*[Circle(radius=0.75,
                                  # stroke_width = 3,
                                  # fill_color = BLACK,
                                  # fill_opacity = GREEN
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
        #
        # title = Tex('Bubble Sort').scale(0.8).move_to(np.array([-5, 3.5, 0]))
        # self.add(title)
        #
        # # 移动方框
        # boxs = VGroup()
        # for i in range(n_circles - 1):
        #     framebox = SurroundingRectangle(circles[i:i + 2], buff=.1, color=BLUE)
        #     boxs.add(framebox)
        #
        # sort_history = VGroup()
        # for item in range(6):
        #     boxs = VGroup()
        #     for i in range(n_circles - 1):
        #         framebox = SurroundingRectangle(circles[i:i + 2], buff=.1, color=BLUE)
        #         boxs.add(framebox)
        #     self.add(boxs[0])
        #     for i in range(len(boxs)):
        #         if texs[i].get_value() > texs[i + 1].get_value():
        #             self.play(Swap(texs[i], texs[i + 1]))
        #             temp = texs[i]
        #             texs[i] = texs[i + 1]
        #             texs[i + 1] = temp
        #         if i <= len(boxs) - 2:
        #             self.play(ReplacementTransform(boxs[i], boxs[i + 1]))
        #     self.play(FadeOut(boxs[-1]))
        #     move_circles_texs = circle_texs.copy()
        #     sort_history.add(move_circles_texs)
        #     self.play(move_circles_texs.animate.scale(0.3).move_to(np.array([0, 3 - 0.4 * item, 0])))
        #
        # brace_out = Brace(sort_history, direction=RIGHT, color=MAROON)
        # text_out = Tex('$m=6$').next_to(brace_out, RIGHT)
        # brace_in = Brace(circle_texs, direction=DOWN, color=MAROON)
        # text_in = Tex('$n=10$').next_to(brace_in, DOWN)
        # self.play(Write(brace_in), Write(text_in), Write(brace_out), Write(text_out))
        # self.wait()
        # self.play(Write(Tex('Time Complexity:$mn$').move_to(np.array([0, -2, 0]))))
        # self.wait()
        # self.play(Write(Tex('if $m=n$: Worst case Time Complexity:$n^2$').scale(0.6).move_to(np.array([0, -2.7, 0]))))
        # self.wait()
        # self.play(Write(Tex('if $m=1$: Best case Time Complexity:$n$').scale(0.6).move_to(np.array([0, -3.4, 0]))))

