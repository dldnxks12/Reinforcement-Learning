### 2023-1 RL study log


- Window GPU Setting - ok
- Linux GPU Setting 

     https://velog.io/@jiyeong3141592/gpusetting - cuda, cuDnn versions for rtx3060 ti  
     https://www.2cpu.co.kr/lec/3998 - cuda install

     - Error Fixing
     https://unix.stackexchange.com/questions/440840/how-to-unload-kernel-module-nvidia-drm  
     https://biology-statistics-programming.tistory.com/158


     - GPU Check  
          - lspci | grep VGA -> GPU Device 
          - nvcc -V or nvcc --version -> CUDA 
          - nvidia-smi -> GPU Env Check
          - cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2 -> cuDnn check 
     

- Reinforcement learning study 

     - RL Baek
     - RL Lee
     - RL Lab book

- Mathematical Knowledges
 
    - Mathematics for Machine Learning
    - Probability and Statistics
    - Linear Algebra
