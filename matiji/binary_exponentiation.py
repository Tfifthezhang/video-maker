from manim import *
import numpy as np

import networkx as nx


class Title(Scene):
    def construct(self):
        text = Text('快速幂算法').scale(1.5)
        self.play(Write(text))
        self.wait(2)


class Power(Scene):
    def construct(self):
        tex_power = MathTex(r'm^n =', r'm \times m \times m \cdots\cdots m \times m \times m')
        # self.add(tex_power)
        self.play(Write(tex_power))

        brace_exp = Brace(tex_power[-1], direction=DOWN, color=MAROON)
        # self.add(brace_exp)
        self.play(Write(brace_exp))

        text_log = Tex('$n$').next_to(brace_exp, DOWN)
        # self.add(text_log)
        self.play(Write(text_log))


class Poly(Scene):
    def construct(self):
        V1_obj = VGroup()
        tex_plus = MathTex(r'n =', r'n_1 + n_2 + n_3 + n_4 + n_5')
        arrow_tex = MathTex(r'\Downarrow')
        tex_power = MathTex(r'm^n =', r'm^{n_1} \times m^{n_2} \times m^{n_3} \times m^{n_4} \times m^{n_5}')

        V1_obj.add(*[tex_plus, arrow_tex, tex_power])
        V1_obj.arrange_submobjects(DOWN, buff=0.1)
        self.play(Write(V1_obj))
        
        self.wait(2)
        self.play(V1_obj.animate.scale(0.8).shift(2.5 * UP))

        V2_obj = VGroup()
        s_tex = [r'm^{n_%d}' % i for i in range(1, 6)]
        V2_obj.add(*[MathTex(i) for i in s_tex])

        V2_obj.arrange_submobjects(RIGHT, buff=0.5)

        #self.play(Transform(V1_obj, V2_obj))
        self.play(Write(V2_obj))

        V3_obj = VGroup()
        V3_obj.add(
            *[CurvedArrow(V2_obj[0].get_center() - [0, 0.3, 0], i.get_center() - [0, 0.3, 0], radius=3, tip_length=0.2)
              for i in V2_obj])
        self.play(Write(V3_obj))
        self.wait(2)
        # self.play(V3_obj.shift(1*UP))

        V4_obj = VGroup()
        V4_obj.add(*[
            CurvedArrow(V2_obj[i].get_center() - [0, 0.2, 0], V2_obj[i + 1].get_center() - [0, 0.2, 0], radius=1,
                        tip_length=0.2) for i in range(4)])
        self.play(Transform(V3_obj, V4_obj))
        self.wait(4)
        

class Binary_Exp(Scene):
    def construct(self):
        title = Text('二进制方法').scale(0.8).move_to(np.array([-5, 3.5, 0]))
        self.add(title)
        
        text_binary = MathTex(r'n = n_t2^t + n_{t-1}2^{t-1} + \cdots\cdots n_12^1 + n_02^0')
        self.play(Create(text_binary))
        
        self.wait(5)
        
        text_exp = MathTex(r'm^n =',r'm^{n_t2^t}',r'\times',r'm^{n_{t-1}2^{t-1}}',r'\times\cdots\cdots', r'm^{n_12^1}',r'\times', r'm^{n_02^0}')
        self.play(Transform(text_binary,text_exp))
        
        framebox = [SurroundingRectangle(text_exp[i], buff = .1) for i in [7,5,3,1]]
        text_iter = [MathTex(r'm_0 = m_0').next_to(framebox[0],DOWN),
                     MathTex(r'm_1 = m_0^2').next_to(framebox[1],DOWN),
                     MathTex(r'm_2 = m_1^2').next_to(framebox[2],DOWN),
                     MathTex(r'm_3 = m_2^2').next_to(framebox[3],DOWN)]
        
        self.play(Create(framebox[0]))
        self.play(Write(text_iter[0]))
        
        self.wait()
        for j in range(3):
            self.play(ReplacementTransform(framebox[j],framebox[j+1]))
            self.play(Write(text_iter[j+1]))
            self.wait()


class Example_Power(Scene):
    def construct(self):
        
        title = Text('例子').scale(0.8).move_to(np.array([-5, 3.5, 0]))
        self.add(title)
        
        V1_obj = VGroup()

        tex_power = MathTex(r'7^{31} =', r'7 \times 7 \times 7 \cdots\cdots 7 \times 7 \times 7')

        brace_exp = Brace(tex_power[-1], direction=DOWN, color=MAROON)

        text_log = Tex('$31$').next_to(brace_exp, DOWN)

        V1_obj.add(*[tex_power, brace_exp, text_log])

        self.add(tex_power, brace_exp, text_log)

        # self.add(V1_obj.scale(0.7).shift(2.5*UP))

        self.play(V1_obj.animate.scale(0.5).shift(3 * UP))

        # V1_obj.arrange_submobjects(DOWN, buff=0.1)

        V2_obj = VGroup()
        tex_binary = MathTex(r'31 = ', r'(11111)_2' ,r'= 1+2+4+8+16')
        arrow_tex = MathTex(r'\Downarrow')
        tex_comb = MathTex(r'7^{31} =', r'7^1 \times 7^2 \times7^4 \times 7^8 \times 7^{16}')

        V2_obj.add(*[tex_binary,arrow_tex,tex_comb])
        V2_obj.arrange_submobjects(DOWN, buff=0.1)
        self.add(V2_obj)

        self.play(Write(V2_obj))
        self.wait(4)
        
        # brace_exp = Brace(tex_binary[1], direction=UP, color=MAROON)
        # text_log = Text('binary').scale(0.5).next_to(brace_exp, UP)
        # self.play(Write(brace_exp,text_log))

        tex_comb2 = MathTex(r'7^{31} =',
                            r'7^1 \times ({7^1})^2 \times ({7^2})^2 \times ({7^4})^2 \times ({7^8})^2').next_to(V2_obj,DOWN)
        # self.add(tex_comb2)

        self.play(Transform(tex_comb, tex_comb2))

        brace_comb = Brace(tex_comb2[-1], direction=DOWN, color=MAROON)
        text_log = MathTex(r'5 = [\log(31)]+1 ').next_to(brace_comb, DOWN)

        self.play(Write(brace_comb))
        self.play(Write(text_log))

        # self.play(Write(V2_obj))
        
        
class python_code(Scene):
    def construct(self):
        codes = VGroup()
        code = ['images/recur_binary.py','images/loop_binary.py']
        
        texts = ['递归法', '循环法']
        d_codes_texts = dict(zip(texts, code))
        for i, j in d_codes_texts.items():
            rendered_code = Code(j, tab_width=2,language="Python", background="window", font="Monospace")
            text = Text(i).next_to(rendered_code, UP)
            codes.add(VGroup(rendered_code,text))
        
        #codes.arrange_submobjects(RIGHT, buff=0.1)
        
        #self.add(codes)
        
        self.play(Create(codes[0]))
        self.wait(10)
        self.play(Transform(codes[0],codes[1]))
        self.wait(10)
        
        
        
class Matiji(Scene):
    def construct(self):
        # self.camera.background_color = WHITE
        title = Text('码蹄集 习题 MT2001').scale(0.9).move_to(np.array([0, 3.5, 0]))
        self.add(title)
        
        image = ImageMobject('images/matiji.jpg').move_to(np.array([0, 0.5, 0]))
        
        self.add(image)
        self.bring_to_back(image)
        
        
