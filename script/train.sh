
### Training

# Please ensure the overfitting experiment can produce reasonable results firstly (using 7 test medical images as training dataset)
python train_csmri.py --solver admm --exp csmri_admm_5x6_12_of --validate_interval 20 --env_batch 12 --rmsize 120 --warmup 20 -lp 0.05 --train_times 5000 --max_step 6 --action_pack 5 -le 0.2


# Then you can train the model on the large-scale setting (using PASCAL VOC dataset)
# Note two GTX-1080Ti GPUs are used in the official implementation
CUDA_VISIBLE_DEVICES=0,1 python train_csmri.py --solver admm --exp csmri_admm_5x6_48 --validate_interval 50 --env_batch 48 --rmsize 480 --warmup 20 -lp 0.05 --train_times 15000 --max_step 6 --action_pack 5 -le 0.2
