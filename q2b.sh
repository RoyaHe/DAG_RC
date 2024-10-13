## pretrain Q2B * NELL 
'''
CUDA_VISIBLE_DEVICES=3 python3 main.py --do_train --do_test --tree_data_path /workspace/KGReasoning_Original/data/NELL-q2b \
--dag_data_path /workspace/DAG-QA/data/NELL/Hard \
 -n 128 -b 512 -d 400 -g 30 --cpu_num 0 --geo box --valid_steps 30000 -lr 0.0001 --max_steps 450001 \
--tasks 1p.2p.3p.2i.3i.up.ip.pi.2u.2s.3s.sp.is.us --cuda \
'''

## test Q2B * NELL * Easy
'''
CUDA_VISIBLE_DEVICES=2 python3 main.py --do_test --tree_data_path /workspace/KGReasoning_Original/data/NELL-q2b \
--dag_data_path /workspace/DAG-QA/data/NELL/Easy \
 -n 128 -b 512 -d 400 -g 24 --cpu_num 0 --geo box --valid_steps 30000 -lr 0.0001 --max_steps 450001 \
--tasks 2s.3s.sp.is.us --cuda --checkpoint_path /workspace/DAG-QA/joint_rel/logs/NELL/1p.2p.3p.2i.3i.up.ip.pi.2u.2s.3s.sp.is.us/box/g-24.0-mode-(none,0.02)/2024.08.27-13:47:35 \
'''

## Pretrain Q2B * FB15k-237
'''
CUDA_VISIBLE_DEVICES=3 python3 main.py --do_train --do_test --tree_data_path /workspace/KGReasoning_Original/data/FB15k-237-betae \
--dag_data_path /workspace/DAG-QA/data/FB15k-237/Hard \
-n 128 -b 512 -d 400 -g 24 --cpu_num 0 --geo box --valid_steps 30000 -lr 0.0001 --max_steps 450001 --box_mode "(none,0.3)" \
--tasks 1p.2p.3p.2i.3i.ip.pi.2u.up.2s.3s.sp.is.us --cuda \
'''

## test Q2B * FB15k-237 * Easy
'''
CUDA_VISIBLE_DEVICES=0 python3 main.py --do_test --tree_data_path /workspace/KGReasoning_Original/data/FB15k-237-betae \
    --dag_data_path /workspace/DAG-QA/data/FB15k-237/Easy \
    -n 128 -b 512 -d 400 -g 16 --box_mode "(none,0.2)" --cpu_num 0 --geo box --valid_steps 15000 -lr 0.0001 --max_steps 450001 \
    --checkpoint_path /workspace/DAG-QA/joint_rel/logs/FB15k-237/1p.2p.3p.2i.3i.ip.pi.2u.up.2s.3s.sp.is.us/box/g-16.0-mode-(none,0.3)/2024.08.27-13:50:13 \
    --tasks 2s.3s.sp.is.us --cuda
'''

## test Q2B * FB15k-237 * Hard
'''
CUDA_VISIBLE_DEVICES=0 python3 main.py --do_test --tree_data_path /workspace/KGReasoning_Original/data/FB15k-237-betae \
--dag_data_path /workspace/DAG-QA/data/FB15k-237/Hard \ 
-n 128 -b 512 -d 400 -g 16 --box_mode "(none,0.2)" --cpu_num 0 --geo box --valid_steps 15000 -lr 0.0001 --max_steps 450001 \
--checkpoint_path /workspace/DAG-QA/joint_rel/logs/FB15k-237/1p.2p.3p.2i.3i.ip.pi.2u.up.2s.3s/box/g-16.0-mode-(none,0.3)/2024.08.26-10:44:31 \
--tasks 2s.3s.sp.is.us --cuda --checkpoint_path
'''

## Pretrain Q2B * FB15k
CUDA_VISIBLE_DEVICES=0 python3 main.py --do_train --do_test --tree_data_path /workspace/KGReasoning_Original/data/FB15k-betae \
 --dag_data_path /workspace/DAG-QA/data/FB15k/Hard -n 128 -b 512 -d 400 -g 24 --cpu_num 0 --geo box --valid_steps 30000 -lr 0.0001 --max_steps 450001 --box_mode "(none,0.34)" \
 --tasks 1p.2p.3p.2i.3i.ip.pi.2u.up.2s.3s.sp.is.us --cuda \


## test Q2B * FB15k * Easy
'''
CUDA_VISIBLE_DEVICES=0 python3 main.py --do_test --tree_data_path /workspace/KGReasoning_Original/data/FB15k-betae \
    --dag_data_path /workspace/DAG-QA/data/FB15k/Easy -n 128 -b 512 -d 400 -g 16 --cpu_num 0 --geo box --valid_steps 30000 -lr 0.0001 --max_steps 450001 --box_mode "(none,0.34)" \
    --tasks 2s.3s.sp.is.us --cuda --checkpoint_path /workspace/DAG-QA/joint_rel/logs/FB15k/1p.2p.3p.2i.3i.ip.pi.2u.up.2s.3s.sp.is.us/box/g-16.0-mode-(none,0.34)/2024.08.27-13:55:31
'''
