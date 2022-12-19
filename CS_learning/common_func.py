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
    def add_shape_object(cls, l_n, color=BLUE, rows=10, cols=10):
        n_circles = len(l_n)

        circles = VGroup(*[Circle(radius=0.5,
                                  color=color
                                  )
                           for _ in range(n_circles)
                           ]
                         )
        circles.arrange_in_grid(rows=rows, cols=cols, buff=0.2)

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
    def add_arrow(cls, start_object, end_object, buff=0, color=RED, max_tip_length_to_length_ratio=0.1):
        arrow = Arrow(start_object, end_object, buff=buff,
                      color=color, max_tip_length_to_length_ratio=max_tip_length_to_length_ratio)
        return arrow

    @classmethod
    def add_table(cls, table_content, row_labels=None, col_labels=None, include_outer_lines=True):
        table = MobjectTable(table=table_content, row_labels=row_labels, col_labels=col_labels,
                             include_outer_lines=include_outer_lines)
        return table

    @classmethod
    def add_function(cls, func, color=BLUE, x_range=(-1, 1)):
        func_graph = FunctionGraph(func, color=color, x_range=x_range)
        return func_graph

    @classmethod
    def add_axes(cls, x_range, y_range, x_length, y_length, axis_config={"include_tip": True, "include_numbers": True},
                 y_axis_config={'scaling': LinearBase()}):
        axes = Axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length, axis_config=axis_config,
                    y_axis_config=y_axis_config)
        return axes

    @classmethod
    def copy_move(cls, l_object, destination, shrinkage=0.5):
        pass