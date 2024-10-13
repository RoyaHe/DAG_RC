## pretrain BetaE * NELL 
CUDA_VISIBLE_DEVICES=1 python3 main.py --cuda --do_train \
--tree_data_path ./data/NELL-betae \
--dag_data_path ./DAG-QA/data/NELL/Hard -n 128 -b 512 -d 400 -g 60 \
-lr 0.0001 --max_steps 450001 --cpu_num 0 --geo beta --valid_steps 30000 \
-betam "(1600,2)" --tasks 1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up.2s.3s.sp.is.us.ins \


## test BetaE * NELL * Easy
CUDA_VISIBLE_DEVICES=3 python3 main.py --cuda --do_test \
  --tree_data_path ./data/NELL-betae \
  --dag_data_path ./DAG-QA/data/NELL/Easy -n 128 -b 512 -d 400 -g 60 \
  -lr 0.0001 --max_steps 450001 --cpu_num 0 --geo beta --valid_steps 30000 \
  -betam "(1600,2)" --tasks 2s.3s.sp.is.us.ins --checkpoint_path \


## test BetaE * NELL * Hard
CUDA_VISIBLE_DEVICES=3 python3 main.py --cuda --do_test \
  --tree_data_path ./data/NELL-betae \
  --dag_data_path ./DAG-QA/data/NELL/Hard -n 128 -b 512 -d 400 -g 60 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo beta --valid_steps 30000 \
  -betam "(1600,2)" --tasks 2s.3s.sp.is.us.ins --checkpoint_path \


## Pretrain BetaE * FB15k-237
CUDA_VISIBLE_DEVICES=3 python3 main.py --cuda --do_train \
  --tree_data_path ./data/FB15k-237-betae \
  --dag_data_path ./DAG-QA/data/FB15k-237/Hard -n 128 -b 512 -d 400 -g 60 \
  -lr 0.0001 --max_steps 450001 --cpu_num 0 --geo beta --valid_steps 30000 \
  -betam "(1800,2)" --tasks 1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up.2s.3s.sp.is.us.ins \

## test BetaE * FB15k-237 * Easy
CUDA_VISIBLE_DEVICES=3 python3 main.py --cuda --do_test \
  --tree_data_path ./data/FB15k-237-betae \
  --dag_data_path ./DAG-QA/data/FB15k-237/Easy -n 128 -b 512 -d 400 -g 60 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo beta --valid_steps 30000 \
  -betam "(1800,2)" --tasks 2s.3s.sp.is.us.ins --checkpoint_path \


## test BetaE * FB15k-237 * Hard
CUDA_VISIBLE_DEVICES=3 python3 main.py --cuda --do_test \
  --tree_data_path ./data/FB15k-237-betae \
  --dag_data_path ./DAG-QA/data/FB15k-237/Hard -n 128 -b 512 -d 400 -g 60 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo beta --valid_steps 30000 \
  -betam "(1800,2)" --tasks 2s.3s.sp.is.us.ins --checkpoint_path \


## Pretrain BetaE * FB15k
CUDA_VISIBLE_DEVICES=0 python3 main.py --cuda --do_train --do_test \
--tree_data_path ./data/FB15k-betae \
--dag_data_path ./DAG-QA/data/FB15k/Hard -n 128 -b 512 -d 400 -g 60 \
-lr 0.0001 --max_steps 450001 --cpu_num 1 --geo beta --valid_steps 30000 \
-betam "(1600,2)" --tasks 1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up.2s.3s.sp.is.us.ins \

## test BetaE * FB15k * Easy
CUDA_VISIBLE_DEVICES=0 python3 main.py --cuda --do_test \
  --tree_data_path ./data/FB15k-betae \
  --dag_data_path ./DAG-QA/data/FB15k/Easy -n 128 -b 512 -d 400 -g 60 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo beta --valid_steps 30000 \
  -betam "(1600,2)" --tasks 2s.3s.sp.is.us.ins --checkpoint_path \
  
## test BetaE * FB15k * Hard
CUDA_VISIBLE_DEVICES=2 python3 main.py --cuda --do_test \
  --tree_data_path ./data/FB15k-betae \
  --dag_data_path ./DAG-QA/data/FB15k/Hard -n 128 -b 512 -d 400 -g 60 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo beta --valid_steps 30000 \
  -betam "(1600,2)" --tasks 2s.3s.sp.is.us.ins --checkpoint_path \
