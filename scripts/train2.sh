cd /mnt/hd1/springc/code/work/vehicle_orientation/patch_img_orientation
python train_model.py main --model_name='resnet34' --bs=192 --val_bs=192 --optimizer='adam' --scheduler='cos' --model_weight='' --epochs=150 --normalize_type=0 --is_fix_lr=False --cuda='1' --data_index=1
