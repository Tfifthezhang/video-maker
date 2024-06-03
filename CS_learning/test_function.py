from manim import *
import sys

sys.path.append('..')
from CS_learning.common_func import CommonFunc


class test(Scene):
    def construct(self):
        output_chart = BarChart(
            values=[0.1] * 2,
            bar_names=['target', 'others'],
            y_range=[0, 1, 10],
            y_length=4,
            x_length=4,
            x_axis_config={"font_size": 50},
        ).scale(0.65).to_edge(UP)
        c_bar_lbls = output_chart.get_bar_labels(font_size=32)

        self.play(FadeIn(output_chart),
                  Create(c_bar_lbls))
        # self.play(Indicate(s[-1]))
