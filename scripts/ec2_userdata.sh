#!/bin/bash
# EC2 bootstrap for pokertell compute runs (Ubuntu 24.04).
# libgles2/libegl1/libopengl0: MediaPipe dlopens GLES at model load even for
# CPU inference; libgl1 is for opencv. ffmpeg decodes AV1 (the cv2 wheel
# cannot) for the transcode step.
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg libgl1 libgles2 libegl1 libopengl0
sudo -u ubuntu bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
sudo -u ubuntu mkdir -p /home/ubuntu/pokertell/data/raw /home/ubuntu/pokertell/data/hands /home/ubuntu/pokertell/data/features
touch /home/ubuntu/bootstrap_base_done
