 #!/bin/bash

function realpath {
  echo "$(cd "$(dirname "$1")"; pwd)/$(basename "$1")"
}

nvidia-docker run -it --rm \
    -v /home/Ying:/root \
    --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
    nvcr.io/nvidia/pytorch:22.01-py3
