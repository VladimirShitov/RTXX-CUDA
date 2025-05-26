# A CUDA Implementation of RTXX *A<sup>t</sup>A* Matrix Multiplication algorithm

# How to install
1. Clone the repo
2. Build the binaries:
```
cmake .
make
```

You might need to install CUDA and set up environment variables before building. This is what worked for me:
```
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$CPATH
```

# How to run

## For single-precision
```
./bin/ata-sp <N> <M> <n_iterations> <check> <recursion_depth>
```

- `N`, `M` – matrix dimensions
- `n_iterations` – how many time to run the algorithms
- `check` – if 1, debugging output will be printed
- `recursion_depth` – how deep the recursion should be

For example:
```
./bin/ata-sp 1024 1024 10 1 3
```

## For double precision
```
./bin/ata-sp <N> <M> <n_iterations> <check> <recursion_depth>
```

## How to setup cutoff?
Set `CUTOFF` in [src/ata.h](src/ata.h)

Original publication:
```
@misc{rybin2025xxtfaster,
      title={$XX^{t}$ Can Be Faster}, 
      author={Dmitry Rybin and Yushun Zhang and Zhi-Quan Luo},
      year={2025},
      eprint={2505.09814},
      archivePrefix={arXiv},
      primaryClass={cs.DS},
      url={https://arxiv.org/abs/2505.09814}, 
}
```