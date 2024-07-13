cd /mnt/hd1/springc/code/work/vehicle_orientation/patch_img_orientation
python train_model.py main --model_name='resnet34' --bs=128 --optimizer='adam' --scheduler='cos' --epochs=150 --normalize_type=0 --cuda='0' --data_index=1
python train_model.py main --model_name='resnet34' --bs=128 --optimizer='adam' --scheduler='cos' --epochs=150 --normalize_type=0 --cuda='0' --data_index=2
