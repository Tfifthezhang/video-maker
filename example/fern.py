from manim import *
import random

config.frame_height = 16
config.frame_width = 9
config.pixel_width = 1080
config.pixel_height = 1920
config.frame_rate = 60

config.background_color = '#455D3E'

class Fern(Scene):
    def construct(self):
        colors = [RED, BLUE, YELLOW, PURPLE]
        ax = Axes(x_range=(-2.7, 2.7), y_range=(0, 10), x_length=8, y_length=10).to_edge(DOWN)
        Dot.set_default(radius=0.03, color=WHITE)
        coords = [(0, 0)]
        dots = [Dot(ax.c2p(0, 0))]
        categories = [None, 1, 1, 3, 1, 2, 1, 1, 3, 1]

        def get_point_from_category(x, y, c):
            if c == 0:
                return 0.00 * x + 0.00 * y, 0.00 * x + 0.16 * y + 0.00
            if c == 1:
                return 0.85 * x + 0.04 * y, -0.04 * x + 0.85 * y + 1.60
            if c == 2:
                return 0.20 * x - 0.26 * y, 0.23 * x + 0.22 * y + 1.60
            if c == 3:
                return -0.15 * x + 0.28 * y, 0.26 * x + 0.24 * y + 0.44
            return None

        for n in range(20000 - 9):
            r = random.random()
            if r < 0.01:
                c = 0
            elif r < 0.86:
                c = 1
            elif r < 0.93:
                c = 2
            else:
                c = 3
            categories.append(c)

        for cat in categories[1:]:
            x, y = get_point_from_category(*coords[-1], cat)
            coords.append((x, y))
            dots.append(Dot(ax.c2p(x, y), color=colors[cat]))

        label = Text(
            "Drawing a fern ~ math edition",
            font="Bitter",
            disable_ligatures=True
        ).scale_to_fit_width(config.frame_width - 0.5).to_edge(UP, buff=0.25)
        self.play(Write(label))
        expl = Paragraph(
            "1. Draw a point",
            "2. Choose rule for new point w.r.t. given odds",
            "3. Draw new point",
            "4. Repeat (a lot)",
            font="Bitter",
            disable_ligatures=True,
            line_spacing=0.75,
        ).scale_to_fit_width(config.frame_width - 1).next_to(label, DOWN).to_edge(LEFT, buff=0.5)
        self.play(Write(expl[0]))
        self.play(FadeIn(dots[0]), run_time=0.5)
        self.play(Flash(dots[0]))
        self.wait(0.5)
        # first dot drawn

        self.play(Write(expl[1:3]))
        rule1 = MathTex(r"x &\leftarrow 0 \\ y &\leftarrow 0.16 y")
        rule2 = MathTex(r"x &\leftarrow 0.85 x + 0.04 y \\ y &\leftarrow -0.04 x + 0.85 y + 1.6")
        rule3 = MathTex(r"x &\leftarrow 0.2 x - 0.26 y \\ y &\leftarrow 0.23 x + 0.22 y + 1.6")
        rule4 = MathTex(r"x &\leftarrow -0.15 x + 0.28 y \\ y &\leftarrow 0.26 x + 0.24 y + 0.44")
        odds = VGroup(
            Tex(r"1\%", color=RED), Tex(r"85\%", color=BLUE), Tex(r"7\%", color=YELLOW),
            Tex(r"7\%", color=PURPLE),
        ).arrange(RIGHT, buff=2).scale_to_fit_width(config.frame_width - 2).next_to(expl, DOWN, buff=0.5)
        rules = VGroup(rule1, rule2, rule3, rule4).arrange_in_grid(4, 1, cell_alignment=LEFT, buff=1.5).next_to(odds,
                                                                                                                DOWN)
        self.play(Write(rules))
        self.wait(0.5)
        self.play(
            Write(odds),
            AnimationGroup(
                *[rules[i].animate.next_to(odds[i], DOWN, buff=0).scale(0.3).shift(0.25 * UP).set_color(colors[i]) for i
                  in range(4)],
                lag_ratio=0.5
            )
        )

        for it in range(1, 10):
            self.play(
                ShowPassingFlash(
                    SurroundingRectangle(VGroup(rules[categories[it]], odds[categories[it]])),
                ),
                AnimationGroup(FadeIn(dots[it]), Flash(dots[it]), lag_ratio=1)
            )
            if it == 1:
                self.play(Write(expl[3][:9]))
            if it == 9:
                self.play(Write(expl[3][9:]))

        self.wait(0.5)
        self.play(
            AnimationGroup(
                *[FadeIn(d) for d in dots[10:]],
                lag_ratio=1,
                run_time=20
            ),
        )
        self.wait(1)
        self.play(VGroup(*dots).animate.set_color(GREEN))
        self.wait(2)
        self.play(FadeOut(*self.mobjects))
        self.wait(0.5)