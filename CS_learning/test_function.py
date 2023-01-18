from manim import *
import sys

sys.path.append('..')
from CS_learning.common_func import CommonFunc

class test(Scene):
    def construct(self):
        s = VGroup(*[RoundedRectangle(corner_radius=0.5, height=1.5) for i in range(9)])
        s.arrange_submobjects(UP, buff=0.2).scale(0.35)
        self.play(Create(s))
        self.play(Indicate(s[-1]))
