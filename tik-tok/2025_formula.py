from manim import *
import numpy as np

import networkx as nx
from random import shuffle


class Title(Scene):
    pass


class Formula2025(ThreeDScene):
    def construct(self):
        self.before =None
        self.target = None
        self.history = None

        self.source_data()

    def source_data(self):
        l = list(range(1, 10))
        tc = VGroup(*[MathTex('{}^3'.format(i), color=BLUE) for i in l])
        tc.scale(3).arrange_in_grid(rows=3, buff=2)
        self.add(tc)

        plus = MathTex('+').scale(2)
        self.add(plus.copy().move_to((tc[0].get_center()+tc[1].get_center())/2).shift(0.1*DOWN))
        self.add(plus.copy().move_to((tc[1].get_center() + tc[2].get_center()) / 2).shift(0.1 * DOWN))
        self.add(plus.copy().move_to((tc[3].get_center() + tc[4].get_center()) / 2).shift(0.1 * DOWN))
        self.add(plus.copy().move_to((tc[4].get_center()+tc[5].get_center())/2).shift(0.1*DOWN))
        self.add(plus.copy().move_to((tc[6].get_center() + tc[7].get_center()) / 2).shift(0.1 * DOWN))
        self.add(plus.copy().move_to((tc[7].get_center() + tc[8].get_center()) / 2).shift(0.1 * DOWN))

        #self.add(plus.copy().move_to((tc[0].get_center()+tc[3].get_center())/2).shift(0.2 * LEFT))
        self.add(plus.copy().move_to((tc[1].get_center() + tc[4].get_center()) / 2).shift(0.2 * LEFT))
        #self.add(plus.copy().move_to((tc[2].get_center() + tc[5].get_center()) / 2).shift(0.2 * LEFT))
        #self.add(plus.copy().move_to((tc[3].get_center()+tc[6].get_center())/2).shift(0.2 * LEFT))
        self.add(plus.copy().move_to((tc[4].get_center() + tc[7].get_center()) / 2).shift(0.2 * LEFT))
        #self.add(plus.copy().move_to((tc[5].get_center() + tc[8].get_center()) / 2).shift(0.2 * LEFT))


        title_2025 = MathTex('2025', color=RED).scale(5).shift(8*UP)
        self.add(title_2025)

        cube = Cube(side_length=2.5, fill_opacity=0.8, fill_color=BLACK, stroke_color=RED, stroke_width=10).next_to(tc, 8.5*DOWN)
        cube.rotate(angle=-35 * DEGREES, axis=UP)
        self.add(cube)

        title_formula = MathTex('2025=1^3+2^3+3^3+4^3+5^3+6^3+7^3+8^3+9^3').scale(1.2).next_to(cube, 5*DOWN)
        self.add(title_formula)

        svg_image = SVGMobject('svg_icon/bird.svg', fill_color=PINK)
        text = Text('@迷路的小画家', font='SIL-Hei-Med-Jian', color=PINK).scale(0.8).next_to(svg_image, 2*RIGHT)
        s_scg = VGroup(svg_image, text).next_to(title_2025, 2 * DOWN)

        self.add(s_scg)

class CubeExample(ThreeDScene):
    def construct(self):
            #self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)

        axes = ThreeDAxes()
        cube = Cube(side_length=3, fill_opacity=0.6, fill_color=BLACK, stroke_color=RED, stroke_width=10)
        cube.rotate(angle=-35 * DEGREES, axis=UP)
        self.add(cube)

class thanks_end(Scene):
    def construct(self):
        svg_image = SVGMobject('svg_icon/bird.svg', fill_color=RED).scale(2.5)

        text = Text('迷路的小画家', font='SIL-Hei-Med-Jian').scale(1.5).next_to(svg_image, 2.5 * DOWN)

        self.play(SpinInFromNothing(VGroup(svg_image, text)))

