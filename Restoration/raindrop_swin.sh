CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Raindrop/RainDS.json --model_name swin --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Raindrop --size 512 512 --resize_size 512 --imagenet_norm

CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Raindrop/RainDS_real.json --model_name swin --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Raindrop --size 512 512 --resize_size 512 --imagenet_norm

CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Raindrop/Raindrop.json --model_name swin --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Raindrop --size 512 512 --resize_size 512 --imagenet_norm
