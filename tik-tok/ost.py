from manim import *
import numpy as np

import networkx as nx
from random import shuffle


class Title(Scene):
    pass

class OST(Scene):
    def construct(self):
        l_offer = None
        self.create_offer()

    def create_offer(self):
        svg_offer = SVGMobject('svg_icon/house_price.svg',fill_color=WHITE).scale(1)
        offer_group = VGroup(*[svg_offer.copy() for _ in range(10)])
        offer_group.arrange_submobjects(DOWN, buff=0.45).scale(0.8)
        self.play(FadeIn(offer_group))
        self.wait(1)

        arrow = Arrow(start=offer_group[0].get_top()+2*RIGHT,
                      end=offer_group[-1].get_bottom()+2*RIGHT,
                      color=MAROON, max_stroke_width_to_length_ratio=50)

        self.play(Write(arrow))

        for i in range(10):
            self.play(Indicate(offer_group[i], color=MAROON))
            self.wait(1)
            self.play(FadeOut(offer_group[i]))

class thanks_end(Scene):
    def construct(self):
        svg_image = SVGMobject('svg_icon/bird.svg', fill_color=MAROON).scale(2.5)

        text = Text('迷路的小画家', font='SIL-Hei-Med-Jian').scale(1.5).next_to(svg_image, 2.5 * DOWN)

        self.play(SpinInFromNothing(VGroup(svg_image, text)))

