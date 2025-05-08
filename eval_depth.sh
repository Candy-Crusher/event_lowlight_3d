# root_dir="/mnt/sdc/yifei/code/3d/results"
# experiment_name='MonST3R_EventControl_LowLight_NoFusion_wCA'
# experiment_name="MonST3R_EventControl_LowLight_SNRFusion"
root_dir="/mnt/sdc/xswu/3d/code/results"
# experiment_name="MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt_mvsec_rect_wE_SNRmultiatt"
# experiment_name="MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt_mvsec_rect_wE_multiatt"
# experiment_name="MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt_mvsec_rect_wE_SNRatt"
# experiment_name="MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt_mvsec_rect_wE_singlefusion"
# experiment_name="MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt_mvsec_rect_wE_singlefusionatt"
# experiment_name="MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt_mvsec_rect_woE"
# experiment_name="MonST3R_PO-TA-S-baseline_rect_inpainted"
# experiment_name="MonST3R_PO-TA-S-baseline_rect_interpolated"
experiment_name="baseline"
    # --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"   \
    # --pretrained="$root_dir/$experiment_name/checkpoint-best.pth"   \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=1 --master_port=29605 launch.py --mode=eval_depth  \
    --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --eval_dataset=mvsec \
    --seq_list="outdoor_night/outdoor_night1" \
    --output_dir="/mnt/sdc/xswu/3d/code/results/${experiment_name}_not50" \
    # --use_event_control \
    # --use_lowlight_enhancer \
    # --event_enhance_mode="easy" \
    # To use the ground truth dynamic mask for davis, add: --use_gt_mask
    # --pretrained="/mnt/sdc/xswu/3d/code/results/$experiment_name/checkpoint-best.pth"   \
    # --eval_dataset=mvsec --output_dir="/mnt/sdc/xswu/3d/code/results/$experiment_name" 
