from manim import *

class Basicpoly(Scene):
    def construct(self):
        cir = Circle(fill_opacity=1)
        line = TangentLine(cir,alpha=1.00,length=5)
        rolling_trace = VGroup(cir, line)
        self.play(Rotate(rolling_trace,angle=4*PI,run_time=8))
        self.wait()

class TracedPathExample(Scene):
    def construct(self):
        circ = Circle(color=RED).shift(4*LEFT)
        dot = Dot(color=RED).move_to(circ.get_start())
        rolling_circle = VGroup(circ, dot)
        trace = TracedPath(circ.get_start)
        rolling_circle.add_updater(lambda m: m.rotate(-0.3))
        self.add(trace, rolling_circle)
        self.play(rolling_circle.animate.shift(8*RIGHT), run_time=4, rate_func=linear)
        
        
        