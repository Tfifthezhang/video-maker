from manim import *
import sys

sys.path.append('..')
from CS_learning.common_func import CommonFunc

class SineWaveFromUnitCircle(Scene):
    def construct(self):
        # 创建坐标系
        axes = Axes(
            x_range=[0, 2 * PI],  # x轴范围 [0, 2π]
            y_range=[-1.5, 1.5],  # y轴范围 [-1.5, 1.5]
            axis_config={"color": BLUE},
        )

        # 绘制单位圆
        unit_circle = Circle(radius=1, color=WHITE)
        unit_circle.move_to(ORIGIN)  # 将单位圆的圆心放到原点

        # 创建一个点，初始化时在单位圆上
        moving_point = Dot(color=RED).move_to(unit_circle.point_at_angle(0))

        # 创建一个图形来表示正弦波
        sine_wave = axes.plot(lambda x: np.sin(x), color=YELLOW)

        # 设置旋转点动画
        angle_tracker = ValueTracker(0)  # 角度跟踪器

        # 动画：单位圆上的点随角度变化旋转
        point_animation = UpdateFromAlphaFunc(
            moving_point, lambda m, dt: m.move_to(unit_circle.point_at_angle(angle_tracker.get_value()))
        )

        # 动画：绘制正弦波
        sine_wave_animation = axes.plot(lambda x: np.sin(x), color=YELLOW)

        # 让点随着角度的变化在单位圆上转动
        self.play(Create(axes), Create(unit_circle), Create(sine_wave))
        self.play(
            Create(moving_point),
            angle_tracker.animate.set_value(2 * PI),
            run_time=6,  # 设置动画时长
            rate_func=linear
        )

        # 播放动画
        self.wait()

class SineWaveAnimation(Scene):
    def construct(self):
        # 创建坐标系
        axes = Axes(
            x_range=[-1, 2 * PI + 1, PI / 4],
            y_range=[-1.5, 1.5, 1],
            axis_config={"color": BLUE},
            tips=False,
        )
        self.add(axes)

        # 创建单位圆
        circle = Circle(radius=1, color=WHITE)
        circle.move_to(axes.c2p(0, 0))
        self.add(circle)

        # 创建单位圆上的点
        dot = Dot(color=RED)
        dot.move_to(circle.get_start())
        self.add(dot)

        # 创建正弦波曲线
        sine_curve = axes.plot(lambda x: np.sin(x), x_range=[0, 2 * PI], color=YELLOW)
        self.add(sine_curve)

        # 创建追踪点
        tracker_dot = Dot(color=RED)
        tracker_dot.move_to(axes.c2p(0, np.sin(0)))
        self.add(tracker_dot)

        # 动画部分
        angle = ValueTracker(0)
        dot.add_updater(lambda m: m.move_to(circle.point_from_proportion(angle.get_value() / (2 * PI))))
        tracker_dot.add_updater(lambda m: m.move_to(axes.c2p(angle.get_value(), np.sin(angle.get_value()))))

        self.play(angle.animate.set_value(2 * PI), run_time=4, rate_func=linear)
        self.wait()

class SineWave(Scene):
    def construct(self):
        # 创建一个坐标轴
        axes = Axes(
            x_length=4,
            y_length=4,
            x_decimal=0,
            y_decimal=0,
            axis_config={"color": BLUE},
        )
        self.add(axes)

        # 创建单位圆
        unit_circle = Circle(radius=1, color=RED)
        self.add(unit_circle)

        # 创建一个点在单位圆上移动
        dot = Dot(color=GREEN)
        self.add(dot)

        # 正弦函数曲线
        sine_graph = axes.plot(lambda x: np.sin(x), x_range=[-2 * np.pi, 2 * np.pi], color=PURPLE)
        self.add(sine_graph)

        # 动画：点在圆上移动
        self.play(
            dot.animate.move_to(unit_circle.point_from_proportion(0)),
            rate_func=linear,
            run_time=2,
        )

        # 动画：点在圆上移动，形成正弦波
        for angle in np.arange(0, 2 * np.pi, 0.01):
            dot.move_to(unit_circle.point_from_proportion(angle))
            self.wait(0.01)  # 控制动画速度

        # 保持最后的动画状态
        self.wait(1)

