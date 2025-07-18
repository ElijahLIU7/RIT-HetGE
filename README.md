# Residue Interaction Type-Aware Heterogeneous Graph Embedding model (RIT-HetGE)

![Architecture of the Heterogeneous Graph Protein Residue Interaction Fusion Network model (HG-RIFN)](framwork.jpg)
## Overview

In this project, we provide a specific implementation of HG-RIFN in Pytorch. The organizational structure of this code repository is as follows:

+ ```EXP/``` stores all baseline algorithms for solving graph classification and regression problems, such as GCN.py (GCN code), DHNE.py (DHNE code) and so on.

## Data preprocessing

For the specific [HRIN-ProTstab](https://huggingface.co/datasets/ElijahL/HRIN-ProTstab) dataset download, please click on the URL. Place the downloaded dataset in path ```data/HRIN-ProTstab``` to complete the algorithm implementation.

Run the ```data/main_reg.py``` file, which is a script file used to query ***pLDDT*** for Alphafold3-predicted protein structures. The following options are supported:

```bash
python data/HRIN-ProTstab/preprocess/pLDDT.py [--dataset]
```

Next, run ```dataset/RIN_generation.py``` file to create data that matches the model input.

```bash
python dataset/RIN_generation.py [--output_dir] [--data_type] [--num_cross]
```

## RUN

Run the ```main_reg.py``` file, which is the main script file used to ***train*** the protein thermal stability **regression** model. The following options are supported:
```bash
python script/main_reg.py [--input] [--results] [--data] [n_trials] [--cuda]
```

Run the ```main_cls.py``` file, which is the main script file used to ***train*** the protein thermal stability **classification** model. The following options are supported:
```bash
python script/main_cls.py  [--input] [--results] [--data] [n_trials] [--cuda]
```
Run the test_cls.py file, which is the main script file used to ***test*** the saved optimal protein thermal stability **regression** model. The following options are supported:
```bash
python script/test_reg.py [--input] [--results] [--data] [--cuda]
```

Run the test_cls.py file, which is the main script file used to ***test*** the saved optimal protein thermal stability **classification** model. The following options are supported:
```bash
python script/test_cls.py [--input] [--results] [--data] [--cuda]
```

## Cite


