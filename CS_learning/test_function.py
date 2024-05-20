from manim import *
import sys

sys.path.append('..')
from CS_learning.common_func import CommonFunc

class test(Scene):
    def construct(self):
        svg_image = SVGMobject('svg_icon/transformer.svg',
                               fill_opacity=None,
                               stroke_color=WHITE,
                               stroke_opacity=1,
                               stroke_width=1).scale(3)
        self.play(Create(svg_image))
        self.wait(2)
        # self.play(Indicate(s[-1]))
