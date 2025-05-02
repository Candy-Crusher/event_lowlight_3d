experiment_name="MonST3R_PO-TA-S-baseline"
# experiment_name="MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt_mvsec_wE"
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29606 launch.py --mode=eval_pose  \
    --pretrained="/mnt/sdc/xswu/3d/code/results/$experiment_name/checkpoint-best.pth"   \
    --eval_dataset=mvsec --output_dir="/mnt/sdc/xswu/3d/code/results/$experiment_name" 
    # To use the ground truth dynamic mask for davis, add: --use_gt_mask
