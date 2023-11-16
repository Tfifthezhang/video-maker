# -*- coding: utf-8 -*-

# Copyright (C) 2021 GRGBanking All Rights Reserved

# @Time    : 2023/2/21 5:09 下午
# @Author  : 张暐（zhangwei）
# @File    : maximum_likelihood.py
# @Email   : zhangwei58@grgbanking.com
# @Software: PyCharm

from manim import *
import numpy as np
import sys
import os

sys.path.append('..')

from CS_learning.common_func import CommonFunc

from sklearn.datasets import make_moons, make_blobs, make_classification
from sklearn import cluster, datasets


class supervised_learning(Scene):
    def construct(self):
        self.basic_formula = None
        self.basic_formula_item = {}

        self.write_formula()

    def write_formula(self):
        formula = MathTex("f", "(", "X", ")", "=", "Y").scale(2.5)
        f, x, y = formula[0], formula[2], formula[-1]
        self.play(Write(formula))
        self.play(x.animate.set_color(BLUE))
        self.wait(1)
        self.play(y.animate.set_color(RED))
        self.wait(1)
        self.play(f.animate.set_color(YELLOW))
        self.wait(2)

        self.basic_formula = formula
        self.basic_formula_item = {'x': x, 'f': f, 'y': y}

        self.play(self.basic_formula.animate.to_edge(UP))
        self.wait(1)

    def unsupervise_example(self):
        n_samples = 500
        seed = 30
        blobs = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=seed)

        self.play(Indicate(self.basic_formula_item['x']))


    def supervise_example(self):
        self.play(Indicate(self.basic_formula_item['x']),
                  Indicate(self.basic_formula_item['y']))
        self.wait(1)


class thanks_end(Scene):
    def construct(self):
        svg_image = SVGMobject('svg_icon/bird.svg', fill_color=MAROON).scale(1.2).to_edge(6 * LEFT + UP)

        text = Text('感谢充电', font='SIL-Hei-Med-Jian').scale(1.2).next_to(svg_image, 2.5 * DOWN)

        self.play(SpinInFromNothing(svg_image))

        self.play(Create(text))

        source_path = 'svg_icon/charge/'

        l_image_path = [image_path for image_path in os.listdir(source_path) if image_path.split('.')[-1] == 'jpg']
        vg_anchor = VGroup(*[Circle(1) for _ in range(len(l_image_path))])
        vg_anchor.arrange_in_grid(rows=3, buff=0.1).to_edge(2 * RIGHT)
        # self.add(vg_anchor)

        for i in range(len(l_image_path)):
            image_path = l_image_path[i]
            split_str = image_path.split('.')
            if split_str[-1] == 'jpg':
                name = split_str[0]
                # image = cv2.imread(os.path.join(source_path, image_path),3)
                # resize_array = cv2.resize(image, (512,512))
                image = ImageMobject(os.path.join(source_path, image_path)).move_to(vg_anchor[i].get_center())
                image.height = 1.2
                image.width = 1.2
                name_text = Text(name).scale(0.6).next_to(image, 0.5 * DOWN)
                self.add(image, name_text)

        self.wait(3)
