CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Raindrop/RainDS.json --model_name nafnet --ckpt_path ./ckpts/nafnet_fulldata_res.ckpt --output_path Raindrop --size 512 512 --resize_size 512

CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Raindrop/RainDS_real.json --model_name nafnet --ckpt_path ./ckpts/nafnet_fulldata_res.ckpt --output_path Raindrop --size 512 512 --resize_size 512

CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Raindrop/Raindrop.json --model_name nafnet --ckpt_path ./ckpts/nafnet_fulldata_res.ckpt --output_path Raindrop --size 512 512 --resize_size 512
