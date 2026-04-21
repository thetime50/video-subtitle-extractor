conda env create --name video-subtitle -f environment.yml
pip install -c "nvidia/label/cuda-11.8.0" cuda-nvcc


conda create --name video-subtitle python=3.12
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install -c conda-forge cudnn=8
conda install --file requirements-1.txt
pip install -r requirements-2.txt
conda env export > video-subtitle.yml


处理过程中发生错误: [Errno 2] No such file or directory: 'E:\\Program Files\\Video Subtitle Extractor\\_internal\\output\\6【硬核】在古代，如何搞一场成功的改革？\\subtitle\\raw.txt'

ffmpeg -i "F:\live\know\渤海小吏\进化缓慢的人性\2 这张照片成为撕裂美国引爆点的传播密码是什么？.mp4" -t 00:05:00 -c copy test.mp4
