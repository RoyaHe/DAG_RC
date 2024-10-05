# DAG_RC
This repository is for implementation for the submission "" to WWW2025. 

------------------------------------

## 1. Dataset Download
All of the datasets are saved in the folder 'Data'. Our datasets are created from three different knowledge graphs in different difficulty test modes. Namely,
- NELL-DAG (Easy) 
- NELL-DAG (Hard)
- FB15k-237-DAG (Easy)
- FB15k-237-DAG (Hard)
- FB15k-DAG (Easy)
- FB15k-DAG (Hard)

## Environment Requirement
- Python 3.7
- PyTorch 1.7
- tqdm

## Reproduce the Results
In our work, we have obtained evaluation results of the following models on NELL-DAG/FB15k-237-DAG/FB15k-DAG datasets under both easy and hard modes:
- Q2B+RC(Comm)/RC(Comm+Distr)/RC(Comm+Distr+Mono)
- BetaE+RC(Comm)/RC(Comm+Distr)/RC(Comm+Distr+Mono)
- ConE+RC(Comm)/RC(Comm+Distr)/RC(Comm+Distr+Mono) 
