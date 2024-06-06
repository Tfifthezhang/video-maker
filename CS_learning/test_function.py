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

        ax = NumberPlane(x_range=[0, 8], y_range=[0, 8], x_length=8, y_length=6,
                         axis_config={"include_tip": False, "include_numbers": False}).scale(0.9).to_edge(2*DOWN+LEFT)
        self.play(Create(ax))

        arrow_1 = Arrow(ax.c2p(0, 0), ax.c2p(1, 2), buff=0)
        tip1_text = Text('北京').scale(0.6).next_to(arrow_1.get_end(), RIGHT)
        vg_1 = VGroup(arrow_1,tip1_text)

        arrow_2 = Arrow(ax.c2p(0, 0), ax.c2p(4, 6), buff=0)
        tip2_text = Text('中国').scale(0.6).next_to(arrow_2.get_end(), RIGHT)
        vg_2 = VGroup(arrow_2, tip2_text)

        self.play(FadeIn(vg_1),
                  FadeIn(vg_2))

        arrow_3 = Arrow(ax.c2p(0, 0), ax.c2p(4, 1), buff=0)
        tip3_text = Text('巴黎').scale(0.6).next_to(arrow_3.get_end(), RIGHT)
        vg_3 = VGroup(arrow_3, tip3_text)

        arrow_4 = Arrow(ax.c2p(0, 0), ax.c2p(7, 5), buff=0)
        tip4_text = Text('法国').scale(0.6).next_to(arrow_4.get_end(), RIGHT)
        vg_4 = VGroup(arrow_4, tip4_text)

        self.play(FadeIn(vg_3),
                  FadeIn(vg_4))

        differ = Arrow(ax.c2p(4, 6), ax.c2p(1, 2), buff=0, color=MAROON)
        differ2 = Arrow(ax.c2p(7, 5), ax.c2p(4, 1), buff=0, color=MAROON)

        self.play(Write(differ))
        self.wait(1)

        self.play(differ.animate.move_to(differ2))

        self.wait(2)

        interface = Text('中国—北京+巴黎=法国').scale(0.8).to_edge(RIGHT)
        self.play(FadeTransform(differ.copy(), interface))
        self.wait(1)
        # self.play(Indicate(s[-1]))
