CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Low-light/LOLv1.json --model_name swin --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Low-light --size 512 512 --resize_size 512 --imagenet_norm

CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Low-light/LOLv2.json --model_name swin --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Low-light --size 512 512 --resize_size 512 --imagenet_norm

CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Low-light/SICE.json --model_name swin --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Low-light --size 512 512 --resize_size 512 --imagenet_norm

CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Low-light/SID.json --model_name swin --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Low-light --size 512 512 --resize_size 512 --imagenet_norm
