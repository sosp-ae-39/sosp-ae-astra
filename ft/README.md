The first time create docker:

1. `sudo bash launch_docker.sh`

2. build fastertransformer

3. install miniconda

4. conda install python=3.8

5. cp -r /opt/conda/lib /root/miniconda/

6. pip install numpy transformers psutil

Benchmarking:

1. ```curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash```
2. ```sudo apt-get install git-lfs```
3. Convert data format:
   ```
   git lfs clone https://huggingface.co/facebook/opt-13b
   python ../examples/pytorch/gpt/utils/huggingface_opt_convert.py \
      -i opt-13b/ \
      -o opt-13b/c-model/ \
      -i_g 1
   ```
4. python run_13b.py
