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

- Q2B + RC(Comm) / RC(Comm+Distr) / RC(Comm+Distr+Mono)

- BetaE + RC(Comm) / RC(Comm+Distr) / RC(Comm+Distr+Mono)

- ConE + RC(Comm) / RC(Comm+Distr) / RC(Comm+Distr+Mono)

Considering the significant number of experiments, we only display the example commands for reproducing our results of Q2B+RC(Comm)/RC(Comm+Distr)/RC(Comm+Distr+Mono) on NELL-DAG Hard dataset here. The commands for the rest experiments are provided in the bash files.

- To reproduce the result of Q2B+RC(Comm) on NELL-DAG Hard, please run the following command

`python3 main.py --do_train --do_test --tree_data_path /workspace/KGReasoning_Original/data/NELL-q2b \
--dag_data_path /workspace/DAG-QA/data/NELL/Hard \
 -n 128 -b 512 -d 400 -g 30 --cpu_num 0 --geo box --valid_steps 30000 -lr 0.0001 --max_steps 450001 \
--tasks 1p.2p.3p.2i.3i.up.ip.pi.2u.2s.3s.sp.is.us --cuda \`

- To reproduce the result of Q2B+RC(Comm+Distr) on NELL-DAG Hard, please run the following command

` `

- To reproduce the result of Q2B+RC(Comm+Distr+Mono) on NELL-DAG Hard, please run the following command

` `


