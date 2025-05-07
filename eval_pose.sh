# experiment_name="MonST3R_PO-TA-S-evlight"
experiment_name="MonST3R_PO-TA-S-baseline_rect"
# experiment_name="MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt_mvsec_rect_woE"
# experiment_name="MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt_mvsec_rect_wE_singlefusionatt"
# experiment_name="MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt_mvsec_wE"
    # --pretrained="/mnt/sdc/xswu/3d/code/results/$experiment_name/checkpoint-best.pth"   \
    # --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"   \
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29611 launch.py --mode=eval_pose  \
    --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --eval_dataset=mvsec --output_dir="/mnt/sdc/xswu/3d/code/results/$experiment_name" \
    --seq_list="outdoor_night/outdoor_night1" \
    --use_event_control \
    # --use_gt_focal \
    # --seq_list="indoor_flying/indoor_flying3" \
    # --seq_list="outdoor_day/outdoor_day2" \
    # To use the ground truth dynamic mask for davis, add: --use_gt_mask
