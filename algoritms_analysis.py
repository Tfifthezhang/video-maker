from manim import *
import numpy as np

import networkx as nx
from random import shuffle


class Title(Scene):
    pass


class HardwareTime(Scene):
    def construct(self):
        l_file = ['内存.svg', '硬盘.svg', 'CPU.svg', 'python.svg']
        l_file_name = ['内存', '硬盘', 'CPU', '编程语言']

        d_files = dict(zip(l_file_name, l_file))
        files = VGroup()
        for i, j in d_files.items():
            image = SVGMobject(file_name="images/" + j,
                               height=1,
                               # stroke_color=WHITE,
                               stroke_width=2,
                               fill_opacity=1)
            text = Text(i).next_to(image, UP)
            file_name_vg = VGroup(image, text)
            files.add(file_name_vg)

        text = Text('程序耗时的来源').next_to(files, 1.5 * UP)

        self.play(Write(text))

        files.arrange_submobjects(RIGHT, buff=1)
        for i in files:
            self.play(Write(i))
            self.wait(4)


class FlattenTime(Scene):
    def construct(self):
        title = Tex(' $[[1, 2, 3], [4, 5, 6], [7], [8, 9]] \Rightarrow [1, 2, 3, 4, 5, 6, 7, 8, 9]$')
        self.add(title)
        self.wait()
        self.play(title.animate.scale(0.7).move_to(np.array([0, 3.5, 0])))

        codes = VGroup()
        code = ['''def func_extend(x):
    out = []
    for sublist in x:
        out.extend(sublist)
    return out
            ''',
                '''def func_for(x):
    return [item for sublist in x for item in sublist]
            ''',
                '''def func_sum_brackets(x):
    return sum(a, [])
            ''',
                '''def func_reduce(x):
    return functools.reduce(operator.concat, x)
            ''']
        texts = ['extend', '双循环', '求和', 'Reduce']
        d_codes_texts = dict(zip(texts, code))
        for i, j in d_codes_texts.items():
            rendered_code = Code(code=j, tab_width=2, background="rectangle",
                                 language="Python", font="Monospace")
            text = Text(i).next_to(rendered_code, UP)

            func = VGroup(text, rendered_code)
            self.play(Create(func))
            self.wait(4)

            self.play(FadeOut(func))

            codes.add(func.scale(0.5))

        codes.arrange_submobjects(DOWN, buff=0.1)
        self.play(Write(codes.shift(4 * LEFT)))
        self.wait()

        image = SVGMobject(file_name="images/" + 'out.svg',
                           height=5,
                           # stroke_color=WHITE,
                           stroke_width=2,
                           fill_opacity=1)
        self.play(Create(image.shift(3 * RIGHT)))
        self.wait(5)

        self.play(Uncreate(codes), Unwrite(title))
        self.play(image.animate.shift(3 * LEFT))
        self.wait()

        brace_x = Brace(image, direction=DOWN, color=MAROON)
        label_x = Text('问题规模 n').scale(0.5).next_to(brace_x, DOWN)
        self.play(Create(brace_x), Write(label_x))
        self.wait(2)

        brace_y = Brace(image, direction=LEFT, color=MAROON)
        label_y = Text('运行时间 T ').scale(0.5).next_to(brace_y, LEFT)
        self.play(Create(brace_y), Write(label_y))


class O_frame(Scene):
    def construct(self):
        texs = VGroup()

        T_tex = MathTex(r'T(n) = 4n^2+2n+2')
        limit_tex = MathTex(r'\lim\limits_{n \to \infty} \frac{2n}{4n^2} =0')
        limit_tex2 = MathTex(r'\lim\limits_{n \to \infty} \frac{2}{4n^2} =0')
        arrow_tex = MathTex(r'\Downarrow')
        O_tex = MathTex(r'T(n)=\mathrm {O} (n^2)')

        texs.add(T_tex, limit_tex, limit_tex2, arrow_tex, O_tex)
        texs.arrange_submobjects(DOWN, buff=0.1)

        self.play(Create(texs))
        self.wait()
        self.play(texs.animate.scale(0.6).shift(3 * RIGHT))


class FrameTimefunction(Scene):
    def construct(self):
        ax = Axes(x_range=[1, 10], y_range=[0, 150, 10],
                  x_length=8, y_length=6,
                  axis_config={"include_tip": True,
                               "include_numbers": True}
                  )

        # labels = ax.get_axis_labels(x_label="x", y_label="y")

        ax.move_to(np.array([-2, 0, 0]))

        def func(x):
            return x ** 2

        def func_linear(x):
            return x

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
            # self.play(FadeOut(texs[:i]), FadeOut(circles[:i]))

        self.wait()
        self.play(FadeOut(point_number))
        self.wait()

        #         l_text_rel.arrange_submobjects(RIGHT)

        #         self.play(l_text_rel.animate.shift(DOWN))
        brace_binary = Brace(l_text_rel, direction=UP, color=MAROON)
        self.play(Write(brace_binary))
        self.wait()
        binary_text = Tex('$ log(10)+1$').next_to(brace_binary, UP)

        self.play(Write(binary_text))
        self.wait()

        self.play(Uncreate(brace_binary), Uncreate(binary_text))
        self.wait()
        self.play(Write(Text('时间复杂度:log(n)').scale(0.7).move_to(np.array([0, -3, 0]))))
        self.wait(5)

        # 插入二叉树
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

        brace_tree = Brace(g_test, direction=RIGHT, color=MAROON)
        self.play(l_text_rel.animate.arrange_submobjects(DOWN, buff=1.2).next_to(brace_tree, RIGHT))
        self.wait()

        self.play(FadeTransform(circle_texs, g_test))
        self.wait()
        self.play(Write(brace_tree))
        self.wait(5)

        # self.play()


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
        # self.add(g_test)
        self.play(Write(g_test))


class BogoSort(Scene):
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
        self.add(circles)

        def greate_texs(positions, l_n):
            texs = VGroup()
            for i in range(10):
                integ = MathTex(l_n[i])
                integ.move_to(positions[i].get_center())
                texs.add(integ)
            return texs

        # texs = greate_texs(circles, l_array)
        # #self.add(texs)

        title = Text('Bogo Sort').scale(0.8).move_to(np.array([-5, 3.5, 0]))
        self.add(title)

        n_loop = 6

        l_texs = VGroup()
        for i in range(n_loop):
            texs = greate_texs(circles, l_array)
            shuffle(l_array)
            l_texs.add(texs)

        l_count = VGroup()
        for i in range(n_loop):
            l_count.add(MathTex(i).move_to(np.array([-5, 3.5, 0])))

        for j in range(len(l_texs) - 1):
            self.wait()
            self.play(TransformMatchingTex(l_texs[j], l_texs[j + 1]))

        # brace_out = Brace(sort_history, direction=RIGHT, color=MAROON)
        # text_out = Tex('$m=6$').next_to(brace_out, RIGHT)
        # brace_in = Brace(circle_texs, direction=DOWN, color=MAROON)
        # text_in = Tex('$n=10$').next_to(brace_in, DOWN)
        # self.play(Write(brace_in), Write(text_in), Write(brace_out), Write(text_out))
        # self.wait()
        # self.play(Write(Tex('Time Complexity:$mn$').move_to(np.array([0, -2, 0]))))
        # self.wait()
        # self.play(Write(Tex('if $m=n$: Worst case Time Complexity:$n^2$').scale(0.6).move_to(np.array([0, -2.7, 0]))))
        # self.wait()
        # self.play(Write(Tex('if $m=1$: Best case Time Complexity:$n$').scale(0.6).move_to(np.array([0, -3.4, 0]))))
