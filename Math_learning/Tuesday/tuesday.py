# -*- coding: utf-8 -*-
from datetime import datetime
import random
from manim import *
import numpy as np
import sys

random.seed(17)
sys.path.append('..')

from CS_learning.common_func import CommonFunc


class TuesdayParadox(Scene):
    def construct(self):
        self.intro_problem()

    def intro_problem(self):
        svg_boy = SVGMobject('svg_icon/boy.svg', fill_color=WHITE)

        self.play(FadeIn(svg_boy))
        self.wait(1)





