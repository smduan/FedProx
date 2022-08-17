# FedProx

This repo is the implementation of the paper [Federated optimization in heterogeneous networks
](https://proceedings.mlsys.org/paper/2020/file/38af86134b65d0f10fe33d30dd76442e-Paper.pdf)

Run this repo:

1. Downloaded datasets and oraganize them as follows:

   ```
   data
   ├── clinical
   │   ├── clinical_test.csv
   │   └── tmp
   │       ├── beta0.05
   │       │   ├── clinical_node_0.csv
   │       │   ├── clinical_node_1.csv
   │       │   ├── clinical_node_2.csv
   │       │   ├── clinical_node_3.csv
   │       │   └── clinical_node_4.csv
   │       ├── beta0.5
   │       └── ...
   └── ...
   ```

2. Edit the configuration file `conf.py`. Some important arguments are:

   - `global_epochs`: number of global epochs
   - `local_epochs`: number of local epochs
   - `beta`: parameter of Dirichlet distribution
   - `mean_batch`: number of instances used in computing the average in FedMix
   - `lambda`: coefficient in loss of FedMix
   - `lr`, `momentum`: optimizer
   - `num_parties`: number of parties
   - ...

3. Start training:

   ```shell
   python main.py
   #or
   python main.py --global_epoch=n --beta=m
   ```
