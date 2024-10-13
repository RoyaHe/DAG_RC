## Train the model on 1p.2p.3p.2i.3i.up.ip.pi.2u.2s.3s.sp.is.us and regularize with 2rs.3rs
CUDA_VISIBLE_DEVICES=3 python3 main.py --do_train --do_test --tree_data_path ./data/NELL-q2b \
    --dag_data_path ./DAG-QA/data/NELL/Hard --asso_path ./DAG-QA/joint_rel/baseline+asso/NELL_Asso --asso_pretrain True \
    --mono_pretrain True \
     -n 128 -b 512 -d 400 -g 18 --cpu_num 0 --geo box --valid_steps 50000 -lr 0.0001 --max_steps 500001 --asso_pretrain_weight 0.00001 \
     --mono_pretrain_weight 0.000001 \
    --tasks 1p.2p.3p.2i.3i.up.ip.pi.2u.2s.3s.sp.is.us --cuda --box_mode "(none,0.06)" \
