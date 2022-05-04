from manim import *
import numpy as np

import networkx as nx


class Title(Scene):
    pass


class FrameTimefunction(Scene):
    def construct(self):
        ax = Axes(x_range=[1, 10], y_range=[0, 150, 10],
                  x_length=8, y_length=6,
                  axis_config={"include_tip": True,
                               "include_numbers": True}
                  )
        # labels = ax.get_axis_labels(x_label="x", y_label="y")

        ax.move_to(np.array([-2, 0, 0]))

        def constant(x):
            return 1

        def func(x):
            return x ** 2

        def func_linear(x):
            return x

        def func_three(x):
            return x ** 3

        def func_log(x):
            return np.log(x) + 1

        def func_linear_log(x):
            return x * np.log(x)

        def func_n_Factorial(x):
            return np.power(x, 6)

        self.play(Create(ax))
        self.wait()

        l_func = [func_log, func_linear, func_linear_log, func, func_n_Factorial]
        texs = ["$\log n$", "$n$", "$n \log n$", "$n^2$", "$n!$"]
        colors = [TEAL, GREEN, YELLOW, GOLD, MAROON]

        for i in range(len(l_func)):
            graph = ax.plot(l_func[i], x_range=[1, 9], color=colors[i], use_smoothing=True)
            graph_label = ax.get_graph_label(graph=graph, label=Tex(texs[i]))
            self.play(Create(graph), Write(graph_label))
            self.wait()


class linear_worst_time_example(Scene):
    def construct(self):

        l_array = [9, 5, 8, 3, 0, 6, 1, 4, 2, 7]
        n_circles = len(l_array)
        circles = VGroup()

        circles = VGroup(*[Circle(radius=0.5,
                                  # stroke_width = 3,
                                  # fill_color = BLACK,
                                  # fill_opacity = GREEN
                                  )
                           for _ in range(n_circles)
                           ]
                         )
        circles.arrange_submobjects(RIGHT, buff=0.3)

        self.add(circles)

        texs = VGroup()
        for i in range(n_circles):
            integ = Integer(number=l_array[i])
            integ.move_to(circles[i].get_center())
            texs.add(integ)

        self.add(texs)

        # 移动向量
        pointer = Vector(DOWN).next_to(circles.get_center(), UP)
        number = Integer(number=7).next_to(pointer, UP)
        point_number = VGroup(pointer, number)
        tracker = ValueTracker(circles[0].get_center()[0])
        point_number.add_updater(lambda x: x.set_x(tracker.get_value()))

        self.add(point_number, tracker)

        # tracker.add_updater(lambda mobject: mobject.width)
        for i in range(10):
            self.play(tracker.animate.set_value(circles[i].get_center()[0]))
            if texs[i].get_value() == number.get_value():
                self.play(Write(Text('True').scale(0.5).move_to(np.array([circles[i].get_center()[0], -2, 0]))))
                self.play(Write(Tex('Worst Case Time Complexity:$O(n)$').move_to(np.array([0, 2.5, 0]))))
            else:
                self.play(Write(Text('False').scale(0.5).move_to(np.array([circles[i].get_center()[0], -2, 0]))))
            self.wait()


class linear_best_time_example(Scene):
    def construct(self):

        l_array = [9, 5, 8, 3, 0, 6, 1, 4, 2, 7]
        n_circles = len(l_array)
        circles = VGroup()

        circles = VGroup(*[Circle(radius=0.5,
                                  # stroke_width = 3,
                                  # fill_color = BLACK,
                                  # fill_opacity = GREEN
                                  )
                           for _ in range(n_circles)
                           ]
                         )
        circles.arrange_submobjects(RIGHT, buff=0.3)

        self.add(circles)

        texs = VGroup()
        for i in range(n_circles):
            integ = Integer(number=l_array[i])
            integ.move_to(circles[i].get_center())
            texs.add(integ)

        self.add(texs)

        # 移动向量
        pointer = Vector(DOWN).next_to(circles.get_center(), UP)
        number = Integer(number=9).next_to(pointer, UP)
        point_number = VGroup(pointer, number)
        tracker = ValueTracker(circles[0].get_center()[0])
        point_number.add_updater(lambda x: x.set_x(tracker.get_value()))

        self.add(point_number, tracker)

        for i in range(1):
            self.play(tracker.animate.set_value(circles[i].get_center()[0]))
            if texs[i].get_value() == number.get_value():
                self.play(Write(Text('True').scale(0.5).move_to(np.array([circles[i].get_center()[0], -2, 0]))))
                self.play(Write(Tex('Best Case Time Complexity:$O(1)$').move_to(np.array([0, 2.5, 0]))))
            else:
                self.play(Write(Text('False').scale(0.5).move_to(np.array([circles[i].get_center()[0], -2, 0]))))
            self.wait()


class BubbleSort(Scene):
    def construct(self):
        l_array = [9, 5, 8, 3, 0, 6, 1, 4, 2, 7]
        n_circles = len(l_array)

        circles = VGroup(*[Circle(radius=0.5,
                                  # stroke_width = 3,
                                  # fill_color = BLACK,
                                  # fill_opacity = GREEN
                                  )
                           for _ in range(n_circles)
                           ]
                         )
        circles.arrange_submobjects(RIGHT, buff=0.3)

        texs = VGroup()
        for i in range(n_circles):
            integ = Integer(number=l_array[i])
            integ.move_to(circles[i].get_center())
            texs.add(integ)

        circle_texs = VGroup(circles, texs)

        self.add(circle_texs)

        title = Text('Bubble Sort').scale(0.8).move_to(np.array([-5, 3.5, 0]))
        self.add(title)

        # 移动方框
        boxs = VGroup()
        for i in range(n_circles - 1):
            framebox = SurroundingRectangle(circles[i:i + 2], buff=.1, color=BLUE)
            boxs.add(framebox)

        sort_history = VGroup()
        for item in range(6):
            boxs = VGroup()
            for i in range(n_circles - 1):
                framebox = SurroundingRectangle(circles[i:i + 2], buff=.1, color=BLUE)
                boxs.add(framebox)
            self.add(boxs[0])
            for i in range(len(boxs)):
                if texs[i].get_value() > texs[i + 1].get_value():
                    self.play(Swap(texs[i], texs[i + 1]))
                    temp = texs[i]
                    texs[i] = texs[i + 1]
                    texs[i + 1] = temp
                if i <= len(boxs) - 2:
                    self.play(ReplacementTransform(boxs[i], boxs[i + 1]))
            self.play(FadeOut(boxs[-1]))
            move_circles_texs = circle_texs.copy()
            sort_history.add(move_circles_texs)
            self.play(move_circles_texs.animate.scale(0.3).move_to(np.array([0, 3 - 0.4 * item, 0])))

        brace_out = Brace(sort_history, direction=RIGHT, color=MAROON)
        text_out = Tex('$m=6$').next_to(brace_out, RIGHT)
        brace_in = Brace(circle_texs, direction=DOWN, color=MAROON)
        text_in = Tex('$n=10$').next_to(brace_in, DOWN)
        self.play(Write(brace_in), Write(text_in), Write(brace_out), Write(text_out))
        self.wait()
        self.play(Write(Tex('Time Complexity:$mn$').move_to(np.array([0, -2, 0]))))
        self.wait()
        self.play(Write(Tex('if $m=n$: Worst case Time Complexity:$n^2$').scale(0.6).move_to(np.array([0, -2.7, 0]))))
        self.wait()
        self.play(Write(Tex('if $m=1$: Best case Time Complexity:$n$').scale(0.6).move_to(np.array([0, -3.4, 0]))))


class BinarySearch(Scene):
    def construct(self):
        l_array = list(range(10))
        n_circles = len(l_array)

        circles = VGroup(*[Circle(radius=0.5,
                                  # stroke_width = 3,
                                  # fill_color = BLACK,
                                  # fill_opacity = GREEN
                                  )
                           for _ in range(n_circles)
                           ]
                         )
        circles.arrange_submobjects(RIGHT, buff=0.3)

        texs = VGroup()
        for i in range(n_circles):
            integ = Integer(number=l_array[i])
            integ.move_to(circles[i].get_center())
            texs.add(integ)

        circle_texs = VGroup(circles, texs)

        self.add(circle_texs)

        title = Text('二分查找(Binary Search)').scale(0.7).move_to(np.array([-4, 3.5, 0]))
        self.add(title)

        pointer = Vector(DOWN).next_to(texs[4].get_center(), UP)
        number = Integer(number=9).next_to(pointer, UP)
        point_number = VGroup(pointer, number)
        self.add(point_number)

        l_text_rel = VGroup()
        for i in [4, 7, 8, 9]:
            self.play(point_number.animate.move_to(np.array([texs[i].get_center()[0], 1, 0])))
            position = texs[i].get_value()
            if number.get_value() > position:
                text_rel = Tex('${}>{}$'.format(number.get_value(), position)).scale(0.6).next_to(point_number, LEFT)
            else:
                text_rel = Tex('${}={}$'.format(number.get_value(), position)).scale(0.6).next_to(point_number, LEFT)
            self.play(Write(text_rel))
            l_text_rel.add(text_rel)
            #self.play(FadeOut(texs[:i]), FadeOut(circles[:i]))



class BinaryTree(Scene):
    def construct(self):
        vertices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        edges = [(4, 1), (4, 7),
                 (1, 0), (1, 2), (7, 5), (7, 8),
                 (2, 3), (5, 6), (8, 9)]

        g = nx.DiGraph()
        g.add_nodes_from(vertices)
        g.add_edges_from(edges)

        g_test = Graph(list(g.nodes), list(g.edges), layout="tree",
                       labels=True,
                       label_fill_color=BLACK,
                       layout_scale=2,
                       root_vertex=4,
                       # vertex_type=Integer,
                       # vertex_config={i: {'number': i} for i in vertices},
                       edge_type=Arrow,
                       edge_config={i: {'max_tip_length_to_length_ratio': 0.01,
                                        'buff': 5}
                                    for i in edges}
        )
        #self.add(g_test)
        self.play(Write(g_test))
