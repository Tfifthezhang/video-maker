from manim import *
import sys

sys.path.append('..')
from CS_learning.common_func import CommonFunc


class Clock(VGroup):
    def __init__(
            self,
            stroke_color: ManimColor = WHITE,
            stroke_width: float = 3.0,
            hour_hand_height: float = 0.3,
            minute_hand_height: float = 0.6,
            tick_length: float = 0.1,
            **kwargs,
    ):
        style = dict(stroke_color=stroke_color, stroke_width=stroke_width)
        circle = Circle(**style)
        ticks = []
        for x, point in enumerate(compass_directions(12, UP)):
            length = tick_length
            if x % 3 == 0:
                length *= 2
            ticks.append(Line(point, (1 - length) * point, **style))
        self.hour_hand = Line(ORIGIN, hour_hand_height * UP, **style)
        self.minute_hand = Line(ORIGIN, minute_hand_height * UP, **style)

        super().__init__(
            circle, self.hour_hand, self.minute_hand,
            *ticks
        )


class test(Scene):
    def construct(self):
        img = ImageMobject(np.uint8([[213, 213, 213, ..., 180, 181, 179],
                                     [213, 213, 213, ..., 180, 181, 179],
                                     [213, 213, 213, ..., 180, 181, 179],
                                     ...,
                                     [192, 186, 176, ..., 194, 194, 193],
                                     [196, 190, 180, ..., 193, 194, 192],
                                     [200, 193, 184, ..., 193, 194, 192]]))
        self.play(Create(img))
        self.wait(1)
        # self.play(Indicate(s[-1]))
