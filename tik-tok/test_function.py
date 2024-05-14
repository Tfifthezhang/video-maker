from manim import *
import sys

sys.path.append('..')
from CS_learning.common_func import CommonFunc

class test(Scene):
    def construct(self):
        tc_text = Text('Time Complexity').scale(0.8)
        tc_tex = Tex('$\\frac{n(n-1)}{2}\sim n^2$').next_to(tc_text, DOWN)

        self.play(Write(tc_text))