CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Motion\ blur/GoPro.json --model_name swin --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Motion_blur --size 512 512 --resize_size 512 --imagenet_norm

CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Motion\ blur/HIDE.json --model_name swin --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Motion_blur --size 512 512 --resize_size 512 --imagenet_norm

CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Motion\ blur/RealBlur.json --model_name swin --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Motion_blur --size 512 512 --resize_size 512 --imagenet_norm

CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Motion\ blur/REDS.json --model_name swin --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Motion_blur --size 512 512 --resize_size 512 --imagenet_norm
