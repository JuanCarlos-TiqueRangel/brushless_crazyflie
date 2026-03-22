FROM osrf/ros:jazzy-desktop

COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /usr/local/bin/

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    git \
    nano \
    libglfw3 \
    libglew2.2 \
    libgl1 \
    libegl1 \
    libglx-mesa0 \
    libosmesa6 \
    libxrender1 \
    libxext6 \
    libxcb-xinerama0 \
    libxcb-cursor0 \
 && rm -rf /var/lib/apt/lists/*

# Install PyTorch first
RUN python3 -m pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Install the rest
RUN python3 -m pip install --no-cache-dir --ignore-installed \
    packaging==25.0 \
    gpytorch \
    ipympl \
    ipywidgets \
    pyserial \
    cflib \
    pymavlink \
    mujoco

WORKDIR /ros_ws
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

CMD ["bash"]