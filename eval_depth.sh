experiment_name="MonST3R_PO-TA-S-baseline"
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29607 launch.py --mode=eval_depth  \
    --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --eval_dataset=mvsec \
    --seq_list="outdoor_night/outdoor_night3" \
    --output_dir="/mnt/sdc/xswu/3d/code/results/$experiment_name" 
    # To use the ground truth dynamic mask for davis, add: --use_gt_mask
    # --pretrained="/mnt/sdc/xswu/3d/code/results/$experiment_name/checkpoint-best.pth"   \
    # --eval_dataset=mvsec --output_dir="/mnt/sdc/xswu/3d/code/results/$experiment_name" 
