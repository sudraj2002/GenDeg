CUDA_VISIBLE_DEVICES=0 python test.py --json_path testjsons/Snow/CSD.json --model_name swin --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Snow --size 512 512 --resize_size 512 --imagenet_norm

CUDA_VISIBLE_DEVICES=0 python test.py --json_path testjsons/Snow/RSVD.json --model_name swin --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Snow --size 512 512 --resize_size 512 --imagenet_norm

CUDA_VISIBLE_DEVICES=0 python test.py --json_path testjsons/Snow/SRRS.json --model_name swin --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Snow --size 512 512 --resize_size 512 --imagenet_norm

CUDA_VISIBLE_DEVICES=0 python test.py --json_path testjsons/Snow/CityscapesSnow.json --model_name swin --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Snow --size 512 512 --resize_size 512 --imagenet_norm

CUDA_VISIBLE_DEVICES=0 python test.py --json_path testjsons/Snow/Snow100k.json --model_name swin --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Snow --size 512 512 --resize_size 512 --imagenet_norm
