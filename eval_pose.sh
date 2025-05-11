root_dir="/mnt/sdc/yifei/code/3d/results"
experiment_name="enhancement_snr_raw_v1_freezeall"
# experiment_name="MonST3R_PO-TA-S-evlight"
# experiment_name="MonST3R_PO-TA-S-baseline_rect"
# experiment_name="MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt_mvsec_rect_woE"
# experiment_name="MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt_mvsec_rect_wE_singlefusionatt"
# experiment_name="MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt_mvsec_wE"
    # --pretrained="/mnt/sdc/xswu/3d/code/results/$experiment_name/checkpoint-best.pth"   \
    # --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"   \
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=29611 launch.py --mode=eval_pose  \
    --pretrained="$root_dir/$experiment_name/checkpoint-best.pth"   \
    --eval_dataset=mvsec \
    --output_dir="$root_dir/$experiment_name" \
    --seq_list="outdoor_night/outdoor_night1" \
    --use_event_control \
    --use_lowlight_enhancer \
    --event_enhance_mode="easy" \
    --event_loss_weight 0.5 \
   --event_loss_start_epoch 0.0 \
   --event_threshold 0.1
    # --use_gt_focal \
    # --seq_list="indoor_flying/indoor_flying3" \
    # --seq_list="outdoor_day/outdoor_day2" \
    # To use the ground truth dynamic mask for davis, add: --use_gt_mask
