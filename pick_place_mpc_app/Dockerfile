FROM osrf/ros:melodic-desktop-full-bionic
MAINTAINER arthur.ho

RUN apt-get update &&  \
    apt-get install -y \
    coinor-libipopt-dev maxima cmake-curses-gui cmake-qt-gui \
    python-catkin-pkg python-catkin-tools libglpk-dev \
    libzmq3-dev libfmt-dev \
    gnome-terminal libcanberra-gtk3-module libcanberra-gtk-module dbus-x11 gdb \
    ros-melodic-rviz ros-melodic-grid-map \
      vim lsb-release curl net-tools iputils-ping mesa-utils less apt-utils valgrind \
 \
      openssh-client \
      libgflags-dev \
 \
      libreadline-dev libssl-dev git gawk \
 \
      socat \
      openjdk-8-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
