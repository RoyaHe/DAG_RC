CUDA_VISIBLE_DEVICES=2 python3 main.py --cuda --do_train --do_test \
    --tree_data_path ./data/NELL-betae \
    --dag_data_path ./DAG-QA/data/NELL/Hard --asso_path ./DAG-QA/NELL_Asso --asso_pretrain True \
    --mono_pretrain True -n 128 -b 512 -d 800 -g 20 --geo cone \
    -lr 0.0001 --max_steps 300001 --cpu_num 0 --valid_steps 60000 --test_batch_size 4 --mono_pretrain_weight 0.00000001 --asso_pretrain_weight 0.00000001\
    --seed 0 --drop 0.2 --tasks 1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up.2s.3s.sp.is.us.ins \
