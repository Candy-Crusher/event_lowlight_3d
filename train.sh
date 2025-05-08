CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29604 launch.py  --mode=train \
    --train_dataset="5_000 @ MVSEC(dset='train', z_far=100, dataset_location='data/MVSEC/processed_rect_odem/outdoor_day/outdoor_day2', S=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[1,2,3,4,5,6,7,8,9], dist_type='linear_1_2', aug_focal=0.9)"   \
    --test_dataset="1811 @ MVSEC(dset='test', z_far=100, dataset_location='data/MVSEC/processed_rect_odem/outdoor_day/outdoor_day2', S=2, strides=[1,2,3,4,5,6,7,8,9], resolution=[(512, 288)], seed=777)"   \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)"  \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)"   \
    --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --lr=5e-5 --min_lr=1e-06 --warmup_epochs=3 --epochs=50 --batch_size=1 --accum_iter=4  \
    --save_freq=3 --keep_freq=5 --eval_freq=1  \
    --output_dir="/mnt/sdc/xswu/3d/code/results/ours_finetune_wE" \
    --wandb \
    --use_event_control \
    --use_lowlight_enhancer \
    --event_enhance_mode="easy" \
    # --use_cross_attention_for_event \
    # --pretrained="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"   \
    # --train_dataset="5_000 @ TarTanAirDUSt3R(dset='train', z_far=80, dataset_location='data/tartanair', S=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[1,2,3,4,5,6,7,8,9], dist_type='linear_1_2', aug_focal=0.9)"   \
    # --test_dataset="1000 @ TarTanAirDUSt3R(dset='test', z_far=80, dataset_location='data/tartanair', S=2, strides=[1,2,3,4,5,6,7,8,9], resolution=[(512, 288)], seed=777)"   \
    # --train_dataset="10_000 @ PointOdysseyDUSt3R(dset='train', z_far=80, dataset_location='data/point_odyssey', S=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[1,2,3,4,5,6,7,8,9], dist_type='linear_1_2', aug_focal=0.9)+ 5_000 @ TarTanAirDUSt3R(dset='Hard', z_far=80, dataset_location='data/tartanair', S=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[1,2,3,4,5,6,7,8,9], dist_type='linear_1_2', aug_focal=0.9)+ 1_000 @ SpringDUSt3R(dset='train', z_far=80, dataset_location='data/spring', S=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[1,2,3,4,5,6,7,8,9], dist_type='linear_1_2', aug_focal=0.9)+ 4_000 @ Waymo(ROOT='data/waymo_processed', pairs_npz_name='waymo_pairs_video.npz', aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, aug_focal=0.9)"   \
    # --test_dataset="1000 @ PointOdysseyDUSt3R(dset='test', z_far=80, dataset_location='data/point_odyssey', S=2, strides=[1,2,3,4,5,6,7,8,9], resolution=[(512, 288)], seed=777)+ 1000 @ SintelDUSt3R(dset='final', z_far=80, S=2, strides=[1,2,3,4,5,6,7,8,9], resolution=[(512, 224)], seed=777)"   \
    # --train_dataset="5_000 @ MVSEC(dset='train', z_far=80, dataset_location='data/MVSEC/processed/outdoor_day/outdoor_day2', S=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[1,2,3,4,5,6,7,8,9], dist_type='linear_1_2', aug_focal=0.9)"   \
    # --test_dataset="1811 @ MVSEC(dset='test', z_far=80, dataset_location='data/MVSEC/processed/outdoor_day/outdoor_day2', S=2, strides=[1,2,3,4,5,6,7,8,9], resolution=[(512, 288)], seed=777)"   \