from manim import *
import sys

sys.path.append('..')
from CS_learning.common_func import CommonFunc


class test(Scene):
    def construct(self):
        cnn = SVGMobject('svg_icon/cnn.svg',
                         opacity=None,
                         stroke_color=BLUE,
                         stroke_opacity=1,
                         stroke_width=1).scale(0.8).to_edge(RIGHT).shift(1.5*UP)

        rnn = SVGMobject('svg_icon/rnn.svg',
                         opacity=None,
                         stroke_color=BLUE,
                         stroke_opacity=1,
                         stroke_width=1).scale(1).next_to(cnn, 2*DOWN)
        #
        self.play(FadeIn(cnn), FadeIn(rnn))

        transformer = SVGMobject('svg_icon/attention.svg',
                                 opacity=None,
                                 stroke_color=BLUE,
                                 stroke_opacity=0.9,
                                 stroke_width=0.8).scale(1.3).to_edge(0.5*LEFT)

        self.play(Create(transformer))
        self.wait(2)
        # self.play(Indicate(s[-1]))
