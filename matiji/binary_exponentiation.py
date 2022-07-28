from manim import *
import numpy as np

import networkx as nx


class Title(Scene):
    def construct(self):
        text = Text('快速幂算法').scale(2)
        self.play(Write(text))
        self.wait(2)


class Power(Scene):
    def construct(self):
        tex_power = MathTex(r'm^n =',r'm \times m \times m \cdots\cdots m \times m \times m')
        self.add(tex_power)
        
        brace_exp = Brace(tex_power[-1], direction=DOWN, color=MAROON)
        self.add(brace_exp)
        
        text_log = Tex('$n$').next_to(brace_exp, DOWN)
        self.add(text_log)
        
class Poly(Scene):
    def construct(self):
        V1_obj = VGroup()
        tex_plus = MathTex(r'n =',r'n_1 + n_2 + n_3 + n_4 + n_5')
        arrow_tex = MathTex(r'\Downarrow')
        tex_power = MathTex(r'm^n =',r'm^{n_1} \times m^{n_2} \times m^{n_3} \times m^{n_4} \times m^{n_5}')
        
        V1_obj.add(*[tex_plus,arrow_tex,tex_power])
        V1_obj.arrange_submobjects(DOWN, buff=0.1)
        self.play(Write(V1_obj))
        
        self.play(V1_obj.animate.scale(0.8).shift(2.5*UP))
        
        
        V2_obj = VGroup()
        s_tex = [r'm^{n_%d}' %i  for i in range(1,6)]
        V2_obj.add(*[MathTex(i) for i in s_tex])
        
        V2_obj.arrange_submobjects(RIGHT, buff=0.5)
        
        self.play(Transform(V1_obj,V2_obj))
        
        V3_obj = VGroup()
        V3_obj.add(*[CurvedArrow(V2_obj[0].get_center()-[0,0.3,0], i.get_center()-[0,0.3,0], radius= 3, tip_length=0.2) for i in V2_obj])
        self.play(Write(V3_obj))
        #self.play(V3_obj.shift(1*UP))
        
        V4_obj = VGroup()
        V4_obj.add(*[CurvedArrow(V2_obj[i].get_center()-[0,0.2,0], V2_obj[i+1].get_center()-[0,0.2,0], radius= 1, tip_length=0.2) for i in range(4)])
        self.play(Transform(V3_obj,V4_obj))
        
class Binary_Exp(Scene):
    def construct(self):
        pass
    
        
        
class Example_Power(Scene):
    def construct(self):
        V1_obj = VGroup()
        
        tex_power = MathTex(r'7^{31} =',r'7 \times 7 \times 7 \cdots\cdots 7 \times 7 \times 7')
        
        brace_exp = Brace(tex_power[-1], direction=DOWN, color=MAROON)
        
        text_log = Tex('$31$').next_to(brace_exp, DOWN)
        
        
        V1_obj.add(*[tex_power,brace_exp,text_log])
        
        self.add(tex_power,brace_exp,text_log)
        
        #self.add(V1_obj.scale(0.7).shift(2.5*UP))
        

        self.play(V1_obj.animate.scale(0.8).shift(2.5*UP))
        
        V2_obj.arrange_submobjects(DOWN, buff=0.1)
        
        
        V2_obj = VGroup()
        tex_binary = MathTex(r'31 = 1+2+4+8+16')
        arrow_tex = MathTex(r'\Downarrow')
        tex_comb = MathTex(r'7^{31} =',r'7^1 \times 7^2 \times7^4 \times 7^8 \times 7^{16}')
        
        V2_obj.add(*[tex_binary,arrow_tex,tex_comb])
        V2_obj.arrange_submobjects(DOWN, buff=0.1)
        self.add(V2_obj)
        
        self.play(Write(V2_obj))
        self.wait(4)
        
        tex_comb2 = MathTex(r'7^{31} =',r'7^1 \times ({7^1})^2 \times ({7^2})^2 \times ({7^4})^2 \times ({7^8})^2').next_to(V2_obj,DOWN)
        #self.add(tex_comb2)
        
        self.play(Transform(tex_comb,tex_comb2))
        
        brace_comb = Brace(tex_comb2[-1], direction=DOWN, color=MAROON)
        text_log = MathTex(r'5 = [\log(31)]+1 ').next_to(brace_comb, DOWN)
        
        self.play(Write(brace_comb))
        self.play(Write(text_log))
        
        #self.play(Write(V2_obj))
        