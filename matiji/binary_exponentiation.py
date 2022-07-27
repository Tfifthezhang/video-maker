from manim import *
import numpy as np

import networkx as nx


class Title(Scene):
    def construct(self):
        text = Text('快速幂算法').scale(2)
        self.play(Write(text))
        self.wait(2)
