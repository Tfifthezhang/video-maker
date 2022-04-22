## 功能介绍

面向个人视频的制作工具，主要用来制作数学、物理和计算机等科普视频

## 使用方法

### docker 使用

~~~shell
docker run -it -v /Users/tfifthefrank/video-maker:/manim/ -p 8888:8888 manimcommunity/manim:stable jupyter lab --ip=0.0.0.0
~~~

-v 表示将宿主机目录/Users/tfifthefrank/video-maker映射到docker目录/manim/ 

-p 表示映射端口

manimcommunity/manim:stable 是我们的镜像名称



