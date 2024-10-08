# [PACT'24] GraNNDis: Fast Distributed Graph Neural Network Training Framework for Multi-Server Clusters [![DOI](https://zenodo.org/badge/821791396.svg)](https://zenodo.org/doi/10.5281/zenodo.12677841)
This repository is the artifact of GraNNDis for PACT'24 artifact evaluation (AE).

Note that this repo provides the SOTA performance on distributed full-batch (full-graph) GNN training even without the GraNNDis schemes from our own optimizations using NCCL.
Our implementations are mainly based on the original code of [PipeGCN](https://github.com/GATECH-EIC/PipeGCN).
For details, please refer to our PACT'24 paper ([Author Copy](PACT24_GraNNDis_Author_Copy.pdf), [Proceeding](https://doi.org/10.1145/3656019.3676892)).

This artifact earned the following badges:
<img width="35%" src="AE/badges.png" align="right"/>
- Artifact Available
- Artifact Evaluated - Reusable
- Results Reproduced

## Getting Started
### 1. SW Dependencies and Setup
- Prerequisite
  - CUDA/CuDNN 11.8 Setting (Make sure to include CUDA paths)
  - Anaconda Setting
  - NFS environment with more than two servers, each server having multiple GPUs.
  - Servers must be accessible by SSH connection (e.g., ssh [user]@[server]).
  ```
  # include the following two lines in ~/.bashrc will include CUDA paths
  export PATH="/usr/local/cuda-11.8/bin:$PATH"
  export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
  nvcc -V # check cuda version using nvcc
  ```
- Main SW Dependencies
  - Python 3.10
  - PyTorch 2.1.0
  - CUDA 11.8
  - DGL 2.1.x

- Setup
  ```
  conda update -y conda # update conda
  conda create -n granndis_ae python=3.10 -y # create conda env
  conda activate granndis_ae # activate conda env
  conda install -c conda-forge -c pytorch -c nvidia -c dglteam/label/th21_cu118 --file conda-requirements.txt -y # install conda packages
  pip install -r pip-requirements.txt # install pip packages
  ```

### 2. HW Dependencies
- Muti-server environment, and each server is equipped with multiple GPUs.
- We need enough system memory (e.g., 256GB) for the artifact evaluation.
- Internal server interconnect (e.g., NVLink) is much faster than external server interconnect (e.g., 10G Ethernet).

### 3. Sample Dataset Preparation & Single Server Test
For the artifact evaluation, we will use three sample datasets (i.e., Arxiv, Reddit, and Products), which are widely used and easily accessible datasets.
We provide a test script (`Codes/brief_masking_test.sh`) to download the datasets and test the created environment.
It will test GraNNDis with default settings.
```
cd Codes
chmod +x brief_masking_test.sh
./brief_masking_test.sh
```
While running the script, you may be required to type `y` for downloading a dataset.
If the tests are successfully conducted, the logs will be saved in `Codes/masking_test_logs/`.
The expected shell results are as follows:
```
... (Omitted) ....
Process 000 | Epoch 00099 | Time(s) 0.0145 | Comm(s) 0.0049 | Reduce(s) 0.0000 | Loss 0.1598
json logs will be saved in  ./masking_test_logs
... (Omitted) ....
============================== Speed result summary ==============================
train_duration :  0.014522718265652657
communication_duration :  0.005530537571758032
reduce duration :  4.990156412532087e-07
loss :  0.5080833435058594
============================================================
... (Omitted) ....
Rank  0 successfully created json log file.
Rank  0 successfully created summary json log file.
... (Omitted) ....
```

## Fast Reproducing of Main Results

Some users may not be familiar with the distributed training procedure, so we provide distributed experiment launchers at `AE/*.py`.
Before reproducing, users must change the configuration fields in the config file (`AE/configs.py`).
```
global_configs = {
    'env_loc': '/nfs/home/ae/anaconda3/envs/granndis_ae/bin/python',
    'runner_loc': '/nfs/home/ae/GraNNDis_Artifact/Codes/main.py',
    'our_runner_loc': '/nfs/home/ae/GraNNDis_Artifact/Codes/our_main.py',
    'workspace_loc': '/nfs/home/ae/GraNNDis_Artifact/',
    'data_loc': '~/datasets/granndis_ae/',
    'num_runners': 2,
    'gpus_per_server': 4,
    'hosts': ['192.168.0.5', '192.168.0.6']
}
```

After modification, just run the following commands, which will show the artifact evaluation results.
```
sh run_ae.sh # run AE scripts
sh parse_ae.sh # parse AE results
```
Then, the results will be saved at `AE*_results.log`.

### Example Test Environment
The example test cluster has two servers, each with four NVIDIA RTX A6000 GPUs.
Internal server GPUs are connected via NVLink Bridge, and servers are connected via 10GbE.

### Expected Trend for the Artifact
As the example test cluster has a 10GbE inter-server connection, the overall speedup could be higher (up to around 6x) than the paper (up to around 3x), which used 32Gbps Infiniband.

All FLX, CoB (with SAGE sampling), and EAS would generally show significant speedup over the baseline optimized full-batch training because GraNNDis minimizes the slow external server communication (AE1). EAS (FLX-EAS) is expected to show more speedup than FLX, especially in larger datasets, such as Products. EAS usually shows higher speedup than CoB (especially in larger datasets) while providing comparable accuracy, as shown in AE2 (accuracy result).

Please note that the result can fluctuate when the inter-server connection is shared with the cluster's NFS file system. In this case, running multiple trials will show the trend mentioned above.

The following are examples of the results of running the above procedure on the authors' remote machine.

### AE1. Throughput Results (Flexible Preloading (FLX), Cooperative Batching (CoB), and Expansion-Aware Sampling (EAS))
The results show that the optimized full-batch training baseline (Opt_FB) suffers from communication overhead, while FLX/CoB addresses such an issue through server-wise preloading.
EAS further accelerates the training through server boundary-aware sampling.
This trend becomes vivid in larger datasets (Reddit and Products).

```
+-------------------------------------------------------+
|              Throughput Results for Arxiv             |
+--------+------------------+-----------------+---------+
| Method | Total Time (sec) | Comm Time (sec) | Speedup |
+--------+------------------+-----------------+---------+
| Opt_FB |      15.40       |       9.85      |   1.00  |
|  FLX   |       8.60       |       2.37      |   1.79  |
|  CoB   |       8.78       |       2.58      |   1.75  |
|  EAS   |      11.67       |       3.70      |   1.32  |
+--------+------------------+-----------------+---------+
+-------------------------------------------------------+
|             Throughput Results for Reddit             |
+--------+------------------+-----------------+---------+
| Method | Total Time (sec) | Comm Time (sec) | Speedup |
+--------+------------------+-----------------+---------+
| Opt_FB |      449.27      |      422.35     |   1.00  |
|  FLX   |      87.55       |      49.67      |   5.13  |
|  CoB   |      90.44       |      49.98      |   4.97  |
|  EAS   |      75.16       |      40.34      |   5.98  |
+--------+------------------+-----------------+---------+
+-------------------------------------------------------+
|            Throughput Results for Products            |
+--------+------------------+-----------------+---------+
| Method | Total Time (sec) | Comm Time (sec) | Speedup |
+--------+------------------+-----------------+---------+
| Opt_FB |      79.67       |      69.15      |   1.00  |
|  FLX   |      20.03       |       6.36      |   3.98  |
|  CoB   |      21.85       |       8.33      |   3.65  |
|  EAS   |      18.23       |       5.78      |   4.37  |
+--------+------------------+-----------------+---------+
```

### AE2. Accuracy Results (Expansion-Aware Sampling (EAS))
As EAS only targets sample server boundary vertices, contributing to acceleration, it successfully achieves comparable accuracy to the original full-batch training.
```
+--------------------------------------+
| Accuracy Comparison (FB vs. FLX-EAS) |
+--------+-------+--------+------------+
| Method | Arxiv | Reddit |  Products  |
+--------+-------+--------+------------+
|   FB   |  0.69 |  0.96  |    0.76    |
|  EAS   |  0.69 |  0.96  |    0.76    |
+--------+-------+--------+------------+
```

## Additional) GraNNDis Arguments & Distributed Launch

For distributed training, we need to set the following common arguments.
```
--n-partitions 4 # set n-partitions as #GPUs per server (for GraNNDis only)
--total-nodes 1 # #servers to conduct training
```

### 1. Optimized Baseline Full-Batch (Full-Graph) Training
The sample argument script for running optimized full-batch training is provided at `Codes/brief_opt_baseline_test.sh`.
The main arguments are as follows:
```
--n-layers 4 # (#conv layers + #linear layers)
--n-linear 1 # (#linear layers)
--model graphsage # model type
--dataset-path /dataset/granndis_ae/ # dataset path
```

### 2. GraNNDis Options

#### Flexible Preloading
The sample argument script for running flexible preloading is provided at `Codes/brief_masking_test.sh`.
The main arguments are as follows:
```
--bandwidth-aware # turn on server-wise preloading
--subgraph-hop 3 # (#conv layers)
--fanout -1 # do not apply sampling and utilize the whole information
--sampler sage # use node-wise sampling
--use-mask # use 1-hop graph masking to support intact full-batch/mini-batch training algorithm
```

#### Cooperative Batching
The sample argument script for running cooperative batching is provided at `Codes/brief_masking_test.sh`.
The main arguments are as follows:
```
--bandwidth-aware # turn on server-wise preloading
--subgraph-hop 3 # (#conv layers)
--fanout 25 # set sage sampling fanout (default: 25)
--sampler sage # use node-wise sampling
--epoch-iter 1 # #iters/epoch, use a larger value if you need finer-grained mini-batch
--use-mask # use 1-hop graph masking to support intact full-batch/mini-batch training algorithm
```

#### Expansion-Aware Sampling
The sample argument script for running expansion-aware sampling is provided at `Codes/brief_sampling_test.sh`.
The main arguments are as follows:
```
--bandwidth-aware # turn on server-wise preloading
--subgraph-hop 1 # sampling hop
--fanout 15 # sampling fanout
--sampler sage # use node-wise sampling
--use-mask # use 1-hop graph masking to express dependency
```

### 3. Distributed Launch
We provide a simple distributed experiment runner interface for users unfamiliar with the distributed launch of training.
The interface utilizes SSH for the distributed launch.
The launchers using this interface are located at `AE/*.py`.
Users can modify this launcher for their own use.


## Citation
```
@inproceedings{song2024granndis,
  title={{GraNNDis}: Fast Distributed Graph Neural Network Training Framework for Multi-Server Clusters},
  author={Song, Jaeyong and Jang, Hongsun and Lim, Hunseong and Jung, Jaewon and Kim, Youngsok and Lee, Jinho},
  booktitle={The 33rd International Conference on Parallel Architectures and Compilation Techniques (PACT 2024)},
  year={2024}
}
```

## License
For the codes from PipeGCN, we follow the license of it (MIT license).
For other codes, the license is also under the MIT license.

## MISC

For a further breakdown of internal/external communication time, users can utilize the `--check-intra-only` option. This option ignores external server communication, so users can figure out the internal server communication time only. The users also can further minimize the one-hop graph masking overhead through removing `--use-mask` option, but it does not provide the intact algorithm.
