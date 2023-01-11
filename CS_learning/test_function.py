from manim import *
import sys

sys.path.append('..')
from CS_learning.common_func import CommonFunc

class test(Scene):
    def construct(self):
        curved_arrow = CommonFunc.add_curvearrow(RIGHT,LEFT)
        self.play(Create(curved_arrow))