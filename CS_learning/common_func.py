from manim import *
import numpy as np


class CommonFunc:
    @classmethod
    def get_integer_shape(cls, l_n, color=BLUE):
        n_circles = len(l_n)

        circles = VGroup(*[Circle(radius=0.5,
                                  color=color
                                  )
                           for _ in range(n_circles)
                           ]
                         )
        circles.arrange_submobjects(RIGHT, buff=0.3)

        texs = VGroup()
        for i in range(n_circles):
            integer = Integer(number=l_n[i])
            integer.move_to(circles[i].get_center())
            texs.add(integer)

        circle_texs = VGroup(circles, texs)

        return circle_texs

    @classmethod
    def get_Integer(cls, l_n, color=BLUE):
        integers = VGroup(*[Integer(number=i,
                                    color=color,
                                    ) for i in l_n
                            ]
                          )
        integers.arrange_submobjects(RIGHT, buff=0.5)
        return integers

    @classmethod
    def pointer_tracker(cls, vg_object, label_name='x', y=0, direction=DOWN, position=UP):
        # 默认在y=0的平面上
        # 一个在上方的向下的箭头
        pointer = Vector(direction).next_to(vg_object[0].get_center(), position)
        # 标签在pointer上方
        label = Text(label_name).add_updater(lambda m: m.next_to(pointer, position))
        # 初始化valuetracker
        tracker = ValueTracker(vg_object[0].get_center()[0])
        # 将 pointer 绑定在 valuetracker上
        pointer.add_updater(lambda m: m.next_to(np.array([tracker.get_value(), y, 0]), position))

        return pointer, tracker, label

    @classmethod
    def variable_tracker(cls, label=Text('x'), start=0, color=RED):
        var = Variable(start, label, var_type=Integer)
        var.set_color(color)
        return var

    @classmethod
    def add_code(cls, code_path, language):
        listing = Code(
            code_path,
            tab_width=4,
            background_stroke_width=1,
            font_size=25,
            background_stroke_color=WHITE,
            line_spacing=0.5,
            insert_line_no=True,
            style=Code.styles_list[15],
            background="window",
            language=language,
        )
        return listing

    @classmethod
    def add_line(cls, start_object, end_object, buff=0):
        line = Line(start_object, end_object, buff=buff)
        return line

    @classmethod
    def add_dashed_line(cls, start_object, end_object, dashed_ratio=0.5):
        dashedline = DashedLine(start_object, end_object, dashed_ratio=dashed_ratio)
        return dashedline

    @classmethod
    def add_arrow(cls, start_object, end_object, buff=0):
        arrow = Arrow(start_object, end_object, buff=buff)
        return arrow


    @classmethod
    def copy_move(cls, l_object, destination, shrinkage=0.5):
        pass