from manim import *
import numpy as np

class Title(Scene):
    pass

class FrameTimefunction(Scene):
    def construct(self):
        ax = Axes(x_range = [1, 10] , y_range =[0, 100, 10],
                  x_length = 8 , y_length = 6,
                  axis_config = {"include_tip": True,
                                 "include_numbers":True}
        )
        labels = ax.get_axis_labels(x_label="x", y_label="y")
        
        ax.move_to(np.array([-2, 0, 0]))
        
        def func(x):
            return x**2
        def func_linear(x):
            return x
        def func_three(x):
            return x**3
        def func_log(x):
            return np.log(x)
        
                
        self.play(Create(ax),Write(labels))
        self.wait()
        
        l_func = [func_log,func_linear,func,func_three]
        texs  = ["$\log n$","$n$","$n^2$","$n^3$"]
        colors = [BLUE,TEAL,GREEN,YELLOW]
        
        for i in range(len(l_func)):
            graph = ax.plot(l_func[i],x_range=[1,9],color=colors[i])
            graph_label = ax.get_graph_label(graph=graph, label=Tex(texs[i]))
            self.play(Create(graph),Write(graph_label))
            self.wait()
            
        
            
class linear_time_example(Scene):
    def construct(self):
        circles = VGroup()
        n_circles = 10
        
        circles  = VGroup(*[Circle(radius = 0.25,
                                   #stroke_width = 3,
                                   #fill_color = BLACK,
                                   #fill_opacity = GREEN
                                  )
                            for _ in range(n_circles)
                           ]
                         )
        circles.arrange_submobjects(RIGHT,buff = 0.2)
        
        self.add(circles)
        
        for i in range(n_circles):
            integ = Tex("${}$".format(i))
            integ.move_to(circles[i].get_center())
            self.add(integ)
            
        # 移动向量
        pointer = Vector(DOWN).next_to(circles.get_center(),UP)
        
        tracker = ValueTracker(circles[0].get_center()[0])
        pointer.add_updater(lambda x : x.set_x(tracker.get_value()))
        
        self.add(pointer,tracker)

        for i in range(1,10):
            self.play(tracker.animate.set_value(circles[i].get_center()[0]))
            self.wait()
        