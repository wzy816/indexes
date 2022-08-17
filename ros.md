# ros

- <http://wiki.ros.org/ROS>
- <https://docs.ros.org/en/galactic/Tutorials.html>

## 读取和解析

- <https://github.com/autovia/ros_hadoop>
  - java, 在 hadoop 中直接处理 bag 的一种方法，定义了  rosbaginputformat, 通过生成  idx.bin 索引作为 spark job 的配置文件
- <https://github.com/event-driven-robotics/importRosbag>
  - python, 对 bag reader 的过程式简化重写
- :star <https://github.com/cruise-automation/rosbag.js>
  - js 版 bag reader/writer, 可解析 message 数据，实现得较完整
- ? <https://github.com/facontidavide/ros_msg_parser>
  - c++,bag inspection tool, alternative for python rqt_plot, rqt_bag and rosbridge
- ? <https://github.com/tu-darmstadt-ros-pkg/cpp_introspection>
  - c++, inspection tool
- <https://github.com/Kautenja/rosbag-tools>
  - python3, 基于 rosbag 库，图片处理、视频处理
- <https://github.com/aktaylor08/RosbagPandas>
  - python2, 基于 rosbag 库提供了读取 rosbag 生成 pandas dataframe 的方法  bag_to_dataframe 生成 dataframe 数据
- <https://github.com/jmscslgroup/bagpy>
  - python, 基于 rosbag 库提供对 bag reader 的 wrapper, 对于预设的 msg 结构简化读取和可视化 msg
- <https://github.com/AtsushiSakai/rosbag_to_csv>
  - python, 基于 rosbag 库基于 read_messages 方法提取 topic，封装了 QtGui
- <https://github.com/IFL-CAMP/tf_bag>
  - python, 基于 rosbag 库提供 BagTfTransformer 类操作 tf 消息，可用 rosdep 安装

## 可视化

- <https://github.com/ToniRV/mesh_rviz_plugins>
  - rviz 插件显示 mesh
- <https://github.com/cruise-automation/webviz>
  - rviz in browser
