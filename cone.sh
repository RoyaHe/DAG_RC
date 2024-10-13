## pretrain ConE * NELL 
'''
CUDA_VISIBLE_DEVICES=0 python3 main.py --cuda --do_train --do_test \
--tree_data_path /workspace/KGReasoning_Original/data/NELL-betae -n 128 -b 512 -d 800 -g 20 --geo cone \
--dag_data_path /workspace/DAG-QA/data/NELL/Hard \
-lr 0.0001 --max_steps 300001 --cpu_num 0 --valid_steps 60000 --test_batch_size 4 \
--seed 0 --drop 0.2 --tasks 1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up.2s.3s.sp.is.us.ins \
'''

## test ConE * NELL * Easy
CUDA_VISIBLE_DEVICES=1 python3 main.py --cuda --do_test \
    --tree_data_path /workspace/KGReasoning_Original/data/NELL-betae -n 128 -b 512 -d 800 -g 20 --geo cone \
    --dag_data_path /workspace/DAG-QA/data/NELL/Easy \
    -lr 0.0001 --max_steps 300001 --cpu_num 0 --valid_steps 60000 --test_batch_size 4 \
    --seed 0 --drop 0.2 --tasks 2s.3s.sp.is.us.ins --checkpoint_path /workspace/DAG-QA/joint_rel/logs/NELL/1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up.2s.3s.sp.is.us.ins/cone/g-20.0-drop-0.2/2024.09.06-12:06:36 \


## Pretrain ConE * FB15k-237
'''
CUDA_VISIBLE_DEVICES=3 python3 main.py --cuda --do_train --do_test \
--tree_data_path /workspace/KGReasoning_Original/data/FB15k-237-betae -n 128 -b 512 -d 800 -g 30 --geo cone \
--dag_data_path /workspace/DAG-QA/data/FB15k-237/Hard \
-lr 0.00005 --max_steps 300001 --cpu_num 0 --valid_steps 30000 --test_batch_size 4 \
--seed 0 --drop 0.1 --tasks 1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up.2s.3s.sp.is.us.ins \
'''
## test ConE * FB15k-237 * Easy
CUDA_VISIBLE_DEVICES=2 python3 main.py --cuda --do_test \
    --tree_data_path /workspace/KGReasoning_Original/data/FB15k-237-betae -n 128 -b 512 -d 800 -g 30 --geo cone \
    --dag_data_path /workspace/DAG-QA/data/FB15k-237/Easy \
    -lr 0.00005 --max_steps 300001 --cpu_num 0 --valid_steps 30000 --test_batch_size 4 \
    --seed 0 --drop 0.1 --tasks 2s.3s.sp.is.us.ins --checkpoint_path /workspace/DAG-QA/joint_rel/logs/FB15k-237/1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up.2s.3s.sp.is.us.ins/cone/g-30.0-drop-0.1/2024.09.06-12:11:01 \


## Pretrain ConE * FB15k
'''
CUDA_VISIBLE_DEVICES=0 python3 main.py --cuda --do_train --do_test \
--tree_data_path /workspace/KGReasoning_Original/data/FB15k-betae -n 128 -b 512 -d 800 -g 40 --geo cone \
--dag_data_path /workspace/DAG-QA/data/FB15k/Hard \
-lr 0.00005 --max_steps 450001 --cpu_num 0 --valid_steps 60000 --test_batch_size 5 \
--seed 0 --drop 0.05 --tasks 1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up.2s.3s.sp.is.us.ins \
'''
## test ConE * FB15k * Easy
CUDA_VISIBLE_DEVICES=0 python3 main.py --do_test \
    --tree_data_path /workspace/KGReasoning_Original/data/FB15k-betae \
    --dag_data_path /workspace/DAG-QA/data/FB15k/Easy -n 128 -b 512 -d 800 -g 40 --drop 0.05 \
    --cpu_num 0 --geo cone --valid_steps 15000 -lr 0.00005 --max_steps 450001 \
    --cuda --tasks 2s.3s.sp.is.us.ins --checkpoint_path  /workspace/DAG-QA/joint_rel/logs/FB15k/1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up.2s.3s.sp.is.us.ins/cone/g-40.0-drop-0.05/2024.09.06-12:11:31\