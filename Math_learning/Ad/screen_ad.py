# -*- coding: utf-8 -*-
from datetime import datetime
import random
from manim import *
import numpy as np
import sys

np.random.seed(17)
sys.path.append('..')

from CS_learning.common_func import CommonFunc


class ScreenFunc(Scene):
    def construct(self):
        self.rec = None
        self.tex = None

        self.intro_screen()
        self.proof1()
        self.proof2()
        #self.curve()

    @staticmethod
    def get_rectangle_corners(bottom_left, top_right):
        return [
            (top_right[0], top_right[1],0),
            (bottom_left[0], top_right[1],0),
            (bottom_left[0], bottom_left[1],0),
            (top_right[0], bottom_left[1],0),
        ]

    def new_rec(self, dia):
        start_point = dia.get_start()
        end_point = dia.get_end()
        res_coor = self.get_rectangle_corners(start_point, end_point)
        polygon = Polygon(*res_coor, color=WHITE)
        return polygon

    def intro_screen(self):

        rect = Rectangle(width=6.0, height=4.0, color=WHITE)

        dia = Line(start=rect.get_corner(DL), end=rect.get_corner(UR), color=MAROON)

        poly_rec = self.new_rec(dia)

        self.play(Create(poly_rec))

        self.wait(2)

        self.rec = VGroup(dia, poly_rec)

        length1 = MathTex('a').scale(0.8).next_to(poly_rec, DOWN)
        length2 = MathTex('b').scale(0.8).next_to(poly_rec, LEFT)

        length1.add_updater(lambda x: x.next_to(poly_rec, DOWN))
        length2.add_updater(lambda x: x.next_to(poly_rec, LEFT))

        self.play(Write(length1),
                  Write(length2))

        len_dia = MathTex('d', '=', '\sqrt{a^2+b^2}', color=MAROON).scale(0.9)
        s_area = MathTex('S', '=', 'ab', color=BLUE).scale(0.9)

        vg_tex = VGroup(len_dia, s_area).arrange_submobjects(DOWN, buff=1).to_edge(RIGHT)

        self.play(SpinInFromNothing(dia))
        self.play(FadeTransform(dia.copy(), vg_tex[0]))

        self.play(poly_rec.animate.set_fill(BLUE, 1))
        self.play(FadeTransform(poly_rec.copy(), vg_tex[1]))
        self.wait(1)

        self.tex = VGroup(length1, length2, len_dia, s_area)

        #t = ValueTracker(0)

        poly_rec.add_updater(lambda x: x.become(self.new_rec(dia)))
        self.play(Rotate(dia, angle=PI, about_point=ORIGIN), run_time=10)
        poly_rec.add_updater(lambda x: x.become(self.new_rec(dia)))

    def proof1(self):
        dia, poly_rec = self.rec

        r =3.6742
        circ = Circle(radius=r, color=PURPLE)
        self.play(FadeIn(circ))

        dot_middle = Dot(ORIGIN, color=BLUE)
        dot_chord = Dot(poly_rec.get_corner(DR), color=BLUE)
        midlle_line = Line(dot_middle.get_center(), dot_chord.get_center(), color=BLUE)

        #midlle_line_pre = Line3D.perpendicular_to(dia, dot_chord.get_center(), color=BLUE)

        def get_projection(line, point):
            d = np.array(line.get_end()-line.get_start())
            v = np.array(point.get_center() - line.get_start())
            return np.array(line.get_start())+np.dot(v, d)/np.dot(d, d)*d

        perpend_line = Line(dot_chord.get_center(), get_projection(dia, dot_chord), color=YELLOW)
        angle = RightAngle(dia, perpend_line,length=0.3,quadrant=(-1,-1))

        self.play(Write(dot_middle),
                  Write(dot_chord),
                  Write(midlle_line))

        self.wait(1)

        self.play(Write(perpend_line),
                  Write(angle))

        self.wait(1)

        dot_chord.add_updater(lambda x: x.become(Dot(poly_rec.get_corner(DR), color=BLUE)))
        midlle_line.add_updater(lambda x: x.become(Line(dot_middle.get_center(), dot_chord.get_center(), color=BLUE)))
        perpend_line.add_updater(lambda x: x.become(Line(dot_chord.get_center(), get_projection(dia, dot_chord), color=YELLOW)))
        angle.add_updater(lambda x: x.become(RightAngle(dia, perpend_line, length=0.3, quadrant=(-1, -1))))

        self.wait(2)

        self.play(Rotate(dia, angle=PI/6, about_point=ORIGIN), run_time=5)

        self.play(Rotate(dia, angle=-PI/9.7, about_point=ORIGIN), run_time=5)

        self.rec.add(VGroup(circ, dot_middle, dot_chord, midlle_line, perpend_line, angle))

    def proof2(self):
        length1, length2, len_dia, s_area = self.tex

        math_eqal = MathTex('a=b').scale(0.8).to_edge(UP+LEFT)
        self.play(FadeTransform(VGroup(length1, length2), math_eqal))

        self.wait(1)

        self.play(self.rec.animate.scale(0.8).to_edge(LEFT))

        math_eqal = MathTex('\sqrt{a\cdot b}' , '\leq','\\frac{a+b}{2}').next_to(self.rec, 1.5*RIGHT)

        self.play(FadeIn(math_eqal))
        self.wait(1)

        math_eqal2 = MathTex('\sqrt{a^2 \cdot b^2}', '\leq', '\\frac{a^2+b^2}{2}').next_to(self.rec, 1.5*RIGHT)

        self.play(ReplacementTransform(math_eqal, math_eqal2))
        self.wait(1)

        self.play(Indicate(s_area))
        self.play(math_eqal2[0].animate.set_color(BLUE))
        self.play(Indicate(len_dia))
        self.play(math_eqal2[-1].animate.set_color(MAROON))

        self.wait(1)

        math_final = MathTex('S^2', '\leq', '\\frac{d^2}{2}').next_to(math_eqal2, DOWN)
        math_final[0].set_color(BLUE)
        math_final[-1].set_color(MAROON)
        self.play(FadeIn(math_final, target_position=math_eqal2))
        self.wait(1)

class RatioFunc(Scene):
    def construct(self):
        self.rec = None
        self.tex = None

        self.display()
        self.curve()

    @staticmethod
    def area_func(x, c=27):
        return x/(x**2+1)*c**2

    def display(self):
        len_dia = MathTex('d', '=', '\sqrt{a^2+b^2}', color=MAROON).scale(0.9)
        s_area = MathTex('S', '=', 'ab', color=BLUE).scale(0.9)
        vg_tex = VGroup(len_dia, s_area).arrange_submobjects(DOWN, buff=1).shift(2*UP)

        self.add(vg_tex)

        ratio_tex = MathTex('x', '=', '\\frac{a}{b}', color=WHITE).scale(0.9). next_to(vg_tex, RIGHT)
        ratio_tex[0].set_color(YELLOW)

        self.play(FadeIn(ratio_tex, target_position=vg_tex))

        sd_tex = MathTex('S', '=', 'f(', 'x', ',','d',')', '=', '\\frac{x}{x^2+1}d^2', color=WHITE). next_to(vg_tex, DOWN)
        sd_tex[0].set_color(BLUE)
        sd_tex[3].set_color(YELLOW)
        sd_tex[3].set_color(MAROON)

        self.play(FadeTransform(vg_tex.copy(), sd_tex))
        self.wait(2)

        self.play(FadeOut(vg_tex),
                  FadeOut(ratio_tex),
                  sd_tex.animate.scale(0.85).to_edge(LEFT+UP))

        self.tex = sd_tex

    def curve(self):
        ax = CommonFunc.add_axes(x_range=[0.5, 3, 0.5], y_range=[0.2, 450, 100], x_length=10, y_length=6,
                                 axis_config={"include_tip": False, "include_numbers": False}).scale(0.8).to_edge(LEFT).shift(0.6*DOWN)
        self.play(Create(ax))
        fit_plot = ax.plot(lambda x: self.area_func(x), x_range=[0.5, 3], use_smoothing=True, color=YELLOW)
        lable = ax.get_graph_label(fit_plot, "d=27", x_val=3, direction=RIGHT)

        self.play(FadeTransform(self.tex.copy(),fit_plot),
                  Write(lable))

        coord1 = ax.c2p(1, self.area_func(1))

        dot_max = Dot(coord1)
        lines_1 = ax.get_lines_to_point(coord1, color=WHITE)

        self.play(Create(dot_max),
                  Create(lines_1))
        self.wait(1)

        rect_16_9 = Rectangle(width=7.1, height=4.0, color=WHITE).scale(0.7)
        size_16_9 = MathTex("x= 16:9", color=MAROON).scale(1)
        rect_3_2 = Rectangle(width=6, height=5.0, color=WHITE).scale(0.7)
        size_3_2 = MathTex("x= 3:2", color=MAROON).scale(1)
        vg_rec = VGroup(rect_16_9, rect_3_2, size_16_9, size_3_2).arrange_in_grid(rows=2).scale(0.6).to_edge(RIGHT).shift(1.5*UP)

        self.play(Create(vg_rec))

        dia_16_9 = Line(start=rect_16_9.get_corner(UL), end = rect_16_9.get_corner(DR), color=YELLOW)
        dia_3_2 = Line(start=rect_3_2.get_corner(UL), end=rect_3_2.get_corner(DR), color=YELLOW)

        self.play(FadeTransform(lable.copy(), dia_16_9),
                  FadeTransform(lable.copy(), dia_3_2))
        self.wait(1)

        coord2 = ax.c2p(1.778, self.area_func(1.778))
        print(self.area_func(1.778))
        dot2 = Dot(coord2)
        lines_2 = ax.get_lines_to_point(coord2, color=WHITE)

        coord3 = ax.c2p(1.5, self.area_func(1.5))
        print(self.area_func(1.5))
        dot3 = Dot(coord3)
        lines_3 = ax.get_lines_to_point(coord3, color=WHITE)

        self.play(FadeTransform(VGroup(rect_16_9, dia_16_9).copy(), VGroup(dot2, lines_2)),
                  size_16_9.copy().animate.next_to(dot2, UP))
        self.wait(1)
        self.play(FadeTransform(VGroup(rect_3_2, dia_3_2).copy(), VGroup(dot3, lines_3)),
                  size_3_2.copy().animate.next_to(dot3, UP))

        # fit_plot2 = ax.plot(lambda x: self.area_func(x, c=28), x_range=[0.5, 10], use_smoothing=True, color=GREEN)
        #
        # self.play(Create(fit_plot2))
        #
        # fit_plot3 = ax.plot(lambda x: self.area_func(x, c=26), x_range=[0.5, 10], use_smoothing=True, color=MAROON)
        #
        # self.play(Create(fit_plot3))

        self.wait(2)

        #lines_1 = ax.get_lines_to_point(circ.get_right(), color=GREEN_B)

class ImageTrans(Scene):
    def construct(self):
        self.intro_image()
    @staticmethod
    def get_image(l_gamma_values):
        import cv2
        def adjust_gamma(image, gamma=1.0):
            # 构建查找表
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
            # 应用查找表
            return cv2.LUT(image, table)
        # 读取图像
        image_path = '/Users/tfifthefrank/Downloads/1901734940339.jpg'  # 替换为你的图像路径
        original_image = cv2.imread(image_path)
        # 显示原始图像和不同伽马值处理后的图像
        for gamma in l_gamma_values:
            adjusted_image = adjust_gamma(original_image, gamma)
            return adjusted_image
    def intro_image(self):
        # from PIL import Image
        # image = Image.open('/Users/tfifthefrank/Downloads/188173493648.jpg')
        # img_array = np.array(image)
        image_array = self.get_image(l_gamma_values=[0.35])
        img = ImageMobject(image_array)
        img.height = 0.8
        self.play(FadeIn(img))
        self.wait(1)