from manim import *
import numpy as np

import networkx as nx


class Title(Scene):
    def construct(self):
        text = Text('快速幂算法').scale(2)
        self.play(Write(text))
        self.wait(2)


class Power(Scene):
    def construct(self):
        text_power = Tex('$m^n = m \cdot m \cdot m \cdots\cdots m \cdot m \cdot m$')
        self.play(Write(text_power))

        self.
