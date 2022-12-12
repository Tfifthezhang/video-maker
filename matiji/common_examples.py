from manim import *
import numpy as np

import networkx as nx


class Title(Scene):
    def construct(self):
        text = Text('快速幂算法').scale(1.5)
        self.play(Write(text))
        self.wait(2)


class number_in_circle(Scene):
    def construct(self):
        l_array = [9, 5, 8, 3, 0, 6, 1, 4, 2, 7]
        n_circles = len(l_array)

        circles = VGroup(*[Circle(radius=0.5,
                                  # stroke_width = 3,
                                  # fill_color = BLACK,
                                  # fill_opacity = GREEN
                                  )
                           for _ in range(n_circles)
                           ]
                         )
        circles.arrange_submobjects(RIGHT, buff=0.3)

        texs = VGroup()
        for i in range(n_circles):
            integ = Integer(number=l_array[i])
            integ.move_to(circles[i].get_center())
            texs.add(integ)

        circle_texs = VGroup(circles, texs)

        self.add(circle_texs)

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
