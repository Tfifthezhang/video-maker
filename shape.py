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
        
class BlockChainExample(Scene):
    def construct(self):
        rect = Rectangle(width=2.0, height=1.0,color=BLUE)
        arrow = Arrow(start=rect.get_right(),end=2*RIGHT,color = GOLD,buff=2/1,stroke_width = 5)
        rect2 = Rectangle(width=2.0, height=1.0,color=BLUE).next_to(arrow)
        vg = VGroup(rect,arrow,rect2)
        #vg_group = VGroup([vg for _ in range(5)])
        self.add(vg)
        
class Chinese(Scene):
    def construct(self):
        text1 = Text('2020年在武汉爆发的新冠疫情虽然很突然\n但中国人的信心没有被摧毁疫情过后所有人都在努力重建生活',font="sans-serif",font_size=20)
        text2 = Text('2022年在上海爆发的新冠疫情摧毁的是信心\\\n尤其是餐饮行业、旅游行业、线下零售行业、电商行业\\\n物流行业、娱乐行业、金融行业，甚至农业和工业.....',font="sans-serif",font_size=20)
        self.play(Transform(text1,text2),run_time=6)
        
class MultipleFonts(Scene):
    def construct(self):
        morning = Text("வணக்கம்", font="sans-serif")
        japanese = Text(
            "日本へようこそ", t2c={"日本": BLUE}
        )  # works same as ``Text``.
        mess = Text("Multi-Language", weight=BOLD)
        russ = Text("Здравствуйте मस नम म ", font="sans-serif")
        hin = Text("नमस्ते", font="sans-serif")
        arb = Text(
            "صباح الخير \n تشرفت بمقابلتك", font="sans-serif"
        )  # don't mix RTL and LTR languages nothing shows up then ;-)
        chinese = Text("臂猿「黛比」帶著孩子", font="sans-serif")
        self.add(morning, japanese, mess, russ, hin, arb, chinese)
        for i,mobj in enumerate(self.mobjects):
            mobj.shift(DOWN*(i-3))
        
        