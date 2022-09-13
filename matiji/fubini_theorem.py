from manim import *
import numpy as np

import networkx as nx


class Title(Scene):
    def construct(self):
        text = Text('数论分块').scale(1.5)
        self.play(Write(text))
        self.wait(2)


class Fubini_question(Scene):
    def construct(self):
        tex_fubini = MathTex(r'\sum_i^n \lfloor \frac{n}{i} \rfloor').scale(1.5)
        # self.add(tex_power)
        self.play(Write(tex_fubini))

        #         brace_exp = Brace(tex_power[-1], direction=DOWN, color=MAROON)
        #         # self.add(brace_exp)
        #         self.play(Write(brace_exp))

        #         text_log = Tex('$n$').scale(1.5).next_to(brace_exp, DOWN)
        #         # self.add(text_log)
        #         self.play(Write(text_log))

        self.wait(2)


class Fubini_function(Scene):
    def get_rectangle_corners(self, bottom_left, top_right):
        return [(top_right[0], top_right[1]),
                (bottom_left[0], top_right[1]),
                (bottom_left[0], bottom_left[1]),
                (top_right[0], bottom_left[1]),
                ]

    def construct(self):
        axes = Axes(x_range=[1, 15, 1],
                    y_range=[0, 15, 1],
                    x_length=8,
                    axis_config={"color": GREEN},
                    x_axis_config={"numbers_to_include": np.arange(1, 15, 1),
                                   "numbers_with_elongated_ticks": np.arange(0, 15, 1)},
                    tips=False,
                    )

        labels = axes.get_axis_labels()

        multiplicative_inverse_graph = axes.plot(lambda x: 15 / x, color=RED)

        multiplicative_inverse_label = axes.get_graph_label(multiplicative_inverse_graph,
                                                            r'y = \lfloor \frac{15}{x} \rfloor',
                                                            direction=UP / 2)

        plot = VGroup(axes, labels, multiplicative_inverse_graph, multiplicative_inverse_label)
        self.play(Write(plot))
        self.play(plot.animate.shift(2 * LEFT))

        l_values = [1, 2, 3, 4, 6, 8]

        def get_rectangle(x, y):
            polygon = Polygon(
                *[axes.c2p(*i) for i in self.get_rectangle_corners((x, 0), (int(15 / y), y))])
            polygon.stroke_width = 1
            polygon.set_fill(BLUE, opacity=1)
            polygon.set_stroke(YELLOW_B)
            return polygon

        tex_group = VGroup()
        text_iter = [r'15 = \lfloor \frac{15}{1} \rfloor',
                     r'7 = \lfloor \frac{15}{2} \rfloor',
                     r'5 = \lfloor \frac{15}{3} \rfloor',
                     r'3 = \lfloor \frac{15}{4} \rfloor',
                     r'2 = \lfloor \frac{15}{6} \rfloor',
                     r'1 = \lfloor \frac{15}{8} \rfloor']
        for i in text_iter:
            tex_group.add(MathTex(i).scale(0.8))

        tex_group.arrange_submobjects(DOWN, buff=0.15)
        tex_group.shift(5 * RIGHT)

        areas = VGroup()
        for t in l_values:
            areas.add(get_rectangle(t, int(15 / t)))

        for s in range(len(l_values)):
            self.play(Create(areas[s]))
            self.play(Write(tex_group[s]))
            self.wait(1)

        trans_tex = [MathTex(r'3 = \lfloor \frac{15}{5} \rfloor').scale(0.8).next_to(tex_group[3], 0 * LEFT),
                     MathTex(r'2 = \lfloor \frac{15}{7} \rfloor').scale(0.8).next_to(tex_group[4], 0 * RIGHT),
                     [MathTex(r'1 = \lfloor \frac{15}{9} \rfloor').scale(0.8).next_to(tex_group[5], 0 * LEFT),
                      MathTex(r'1 = \lfloor \frac{15}{10} \rfloor').scale(0.8).next_to(tex_group[5], 0 * LEFT),
                      MathTex(r'1 = \lfloor \frac{15}{11} \rfloor').scale(0.8).next_to(tex_group[5], 0 * LEFT),
                      MathTex(r'1 = \lfloor \frac{15}{12} \rfloor').scale(0.8).next_to(tex_group[5], 0 * LEFT),
                      MathTex(r'1 = \lfloor \frac{15}{13} \rfloor').scale(0.8).next_to(tex_group[5], 0 * LEFT),
                      MathTex(r'1 = \lfloor \frac{15}{14} \rfloor').scale(0.8).next_to(tex_group[5], 0 * LEFT),
                      MathTex(r'1 = \lfloor \frac{15}{15} \rfloor').scale(0.8).next_to(tex_group[5], 0 * LEFT)]
                     ]

        self.play(Transform(tex_group[3], trans_tex[0]), run_time=2)
        self.play(Transform(tex_group[4], trans_tex[1]), run_time=2)
        for j in trans_tex[-1]:
            self.play(Transform(tex_group[5], j))

        self.wait(2)


class Matiji(Scene):
    def construct(self):
        # self.camera.background_color = WHITE
        title = Text('码蹄集 习题 MT2001').scale(0.9).move_to(np.array([0, 3.5, 0]))
        self.add(title)

        image = ImageMobject('images/matiji.jpg').move_to(np.array([0, 0.5, 0]))

        self.add(image)

        self.wait(5)
        math_tex = MathTex(r'2^k-1=', r'(111...111)_2').scale(1.5).next_to(image, DOWN)
        tex_brace = Brace(math_tex[-1], direction=DOWN, color=MAROON)
        text_log = MathTex(r'k').next_to(tex_brace, DOWN)

        tex_group = VGroup()
        tex_group.add(*[math_tex, tex_brace, text_log])

        self.play(Write(tex_group, run_time=3))


class matiji_code(Scene):
    def construct(self):
        title = Text('关键点：二进制数长度为k，直接遍历').scale(0.9).move_to(np.array([0, 3.5, 0]))
        self.add(title)

        rendered_code = Code('images/matiji_mt2001.py',
                             tab_width=2,
                             language="Python",
                             background="window",
                             font="Monospace")
        self.play(Write(rendered_code))

        self.wait(3)


class screen(Scene):
    def construct(self):
        svg = SVGMobject('images/bird.svg').scale(2)

        text = Text('像小鸟一样努力').next_to(svg, DOWN)

        self.play(SpinInFromNothing(svg))
