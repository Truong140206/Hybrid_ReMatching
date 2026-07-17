for seed in 42
do
python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port='29500' \
        --use_env main.py \
        imr_hideprompt_5e \
        --model vit_base_patch16_224 \
        --original_model vit_base_patch16_224 \
        --batch-size 128 \
        --ca_storage_efficient_method covariance \
        --epochs 20 \
        --data-path /home/ubuntu/wwc/hide1/HiDe-Prompt-main/datasets \
        --lr 0.0005 \
        --ca_lr 0.005 \
        --crct_epochs 30 \
        --seed $seed \
        --train_inference_task_only \
        --output_dir ./output/imr_vit_multi_centroid_mlp_2_seed$seed
done


for seed in 42
do
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29513' \
        --use_env main.py \
        imr_lora \
        --model vit_base_patch16_224 \
        --original_model vit_base_patch16_224 \
        --batch-size 24 \
        --epochs 50 \
        --data-path /home/ubuntu/wwc/hide1/HiDe-Prompt-main/datasets \
        --ca_lr 0.005 \
        --crct_epochs 30 \
        --seed 42 \
        --lr 0.03 \
        --con 0.2 \
        --lora_rank 5 \
        --En gen \
        --tau -10 \
        --K 5 \
        --sched cosine \
        --dataset Split-Imagenet-R \
        --lora_momentum 0.4 \
        --lora_type hide \
        --trained_original_model ./output/imr_vit_multi_centroid_mlp_2_seed$seed \
        --output_dir ./output/test_imr_sup21k_lora_pe_seed42 
done
