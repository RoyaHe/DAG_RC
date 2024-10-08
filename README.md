# DAGE
This repository is for implementation for the submission "DAGE: DAG Query Answering via Relational Combinator with Logical Constraints" to WWW2025. 

------------------------------------

## 1. Datasets Preparation
The datasets include three parts: tree-form queries and DAG queries. The tree-form queries can be accessed via this [link](http://snap.stanford.edu/betae/KG_data.zip). All DAG queries are saved in the folder 'Data'. 
Our datasets are created from three different knowledge graphs in different difficulty test modes. Namely,
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

## Reproduce the Results of DAGE
In our work, we have implemented DAGE on top of Query2Box, BetaE and ConE and evaluated them on NELL-DAG/FB15k-237-DAG/FB15k-DAG datasets under both easy and hard modes:

- Query2Box + DAGE
  
- BetaE + DAGE

- ConE + DAGE

Here we display the commands for reproducing the results of Query2Box/BetaE/ConE+DAGE on NELL-DAG datasets. The rest experiments can be checked in the bash files.

- To reproduce the result of Query2Box+DAGE on NELL-DAG under Hard mode, please run the following command

```
python3 main.py --do_train --do_test --tree_data_path ./tf-data/NELL-q2b --dag_data_path ./Data/NELL/Hard \
-n 128 -b 512 -d 400 -g 30 --cpu_num 0 --geo box --valid_steps 30000 -lr 0.0001 --max_steps 450001 \
--tasks 1p.2p.3p.2i.3i.up.ip.pi.2u.2s.3s.sp.is.us --cuda
```

- To reproduce the result of Query2Box+DAGE on NELL-DAG Easy mode, please run the following command

```
python3 main.py --do_train --do_test --tree_data_path ./tf-data/NELL-q2b --dag_data_path ./Data/NELL/Easy \
-n 128 -b 512 -d 400 -g 30 --cpu_num 0 --geo box --valid_steps 30000 -lr 0.0001 --max_steps 450001 \
--tasks 1p.2p.3p.2i.3i.up.ip.pi.2u.2s.3s.sp.is.us --cuda
```

- To reproduce the result of BetaE+DAGE on NELL-DAG under Hard mode, please run the following command
```
python3 main.py --cuda --do_train --do_test \
--tree_data_path /workspace/KGReasoning_Original/data/NELL-betae \
--dag_data_path /workspace/DAG-QA/data/NELL/Hard -n 128 -b 512 -d 400 -g 60 \
-lr 0.0001 --max_steps 450001 --cpu_num 0 --geo beta --valid_steps 30000 \
-betam "(1600,2)" --tasks 1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up.2s.3s.sp.is.us.ins \
```

- To reproduce the result of BetaE+DAGE on NELL-DAG under Easy mode, please run the following command
```
python3 main.py --cuda --do_train --do_test \
--tree_data_path /workspace/KGReasoning_Original/data/NELL-betae \
--dag_data_path /workspace/DAG-QA/data/NELL/Easy -n 128 -b 512 -d 400 -g 60 \
-lr 0.0001 --max_steps 450001 --cpu_num 0 --geo beta --valid_steps 30000 \
-betam "(1600,2)" --tasks 1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up.2s.3s.sp.is.us.ins \
```

- To reproduce the result of ConE+DAGE on NELL-DAG under Easy mode, please run the following command
```
python3 main.py --cuda --do_train --do_test \
--tree_data_path /workspace/KGReasoning_Original/data/NELL-betae -n 128 -b 512 -d 800 -g 20 --geo cone \
--dag_data_path /workspace/DAG-QA/data/NELL/Hard \
-lr 0.0001 --max_steps 300001 --cpu_num 0 --valid_steps 60000 --test_batch_size 4 \
--seed 0 --drop 0.2 --tasks 1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up.2s.3s.sp.is.us.ins \
```

- To reproduce the result of ConE+DAGE on NELL-DAG under Easy mode, please run the following command
```
python3 main.py --cuda --do_train --do_test \
--tree_data_path /workspace/KGReasoning_Original/data/NELL-betae -n 128 -b 512 -d 800 -g 20 --geo cone \
--dag_data_path /workspace/DAG-QA/data/NELL/Easy \
-lr 0.0001 --max_steps 300001 --cpu_num 0 --valid_steps 60000 --test_batch_size 4 \
--seed 0 --drop 0.2 --tasks 1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up.2s.3s.sp.is.us.ins \
```  

## Additional Constraints

### Distributivity


### Monoticity


