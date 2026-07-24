#!/bin/bash
# EC2 bootstrap for pokertell compute runs (Ubuntu 24.04).
# libgles2/libegl1/libopengl0: MediaPipe dlopens GLES at model load even for
# CPU inference; libgl1 is for opencv. ffmpeg decodes AV1 (the cv2 wheel
# cannot) for the transcode step.
set -e
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg libgl1 libgles2 libegl1 libopengl0 unzip
# AWS CLI v2 via the official installer: the apt package name is not
# reliable across Ubuntu releases, and one bad name kills the whole line.
curl -sL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
cd /tmp && unzip -q -o awscliv2.zip && ./aws/install
sudo -u ubuntu bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
sudo -u ubuntu mkdir -p /home/ubuntu/pokertell/data/raw /home/ubuntu/pokertell/data/hands /home/ubuntu/pokertell/data/features
touch /home/ubuntu/bootstrap_base_done
