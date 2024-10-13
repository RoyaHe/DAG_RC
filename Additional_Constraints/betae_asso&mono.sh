## Train the model on 1p.2p.3p.2i.3i.up.ip.pi.2u.2s.3s.sp.is.us and regularize with 2rs.3rs
CUDA_VISIBLE_DEVICES=2 python3 main.py --cuda --do_train --do_test \
    --tree_data_path ./data/NELL-betae \
    --dag_data_path ./DAG-QA/data/NELL/Hard --asso_path ./DAG-QA/NELL_Distr --asso_pretrain True\
    --mono_pretrain True -n 128 -b 512 -d 400 -g 20 \
    -lr 0.0001 --max_steps 500001 --cpu_num 0 --geo beta --valid_steps 30000 --mono_pretrain_weight 0.00000001 --asso_pretrain_weight 0.00000001\
    -betam "(1600,2)" --tasks 1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up.2s.3s.sp.is.us.ins \
