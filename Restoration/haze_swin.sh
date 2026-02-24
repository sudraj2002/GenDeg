CUDA_VISIBLE_DEVICES=0 python test.py --json_path testjsons/Haze/DenseHaze.json --model_name swin --imagenet_norm --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Haze --size 512 512 --resize_size 512

CUDA_VISIBLE_DEVICES=0 python test.py --json_path testjsons/Haze/RESIDE.json --model_name swin --imagenet_norm  --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Haze --size 512 512 --resize_size 512

CUDA_VISIBLE_DEVICES=0 python test.py --json_path testjsons/Haze/CityscapesFog.json --model_name swin --imagenet_norm  --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Haze --size 512 512 --resize_size 512

CUDA_VISIBLE_DEVICES=0 python test.py --json_path testjsons/Haze/I-HAZE.json --model_name swin --imagenet_norm  --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Haze --size 512 512 --resize_size 512

CUDA_VISIBLE_DEVICES=0 python test.py --json_path testjsons/Haze/NH-HAZE.json --model_name swin --imagenet_norm  --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Haze --size 512 512 --resize_size 512

CUDA_VISIBLE_DEVICES=0 python test.py --json_path testjsons/Haze/O-HAZE.json --model_name swin --imagenet_norm  --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Haze --size 512 512 --resize_size 512

CUDA_VISIBLE_DEVICES=0 python test.py --json_path testjsons/Haze/REVIDE.json --model_name swin --imagenet_norm  --ckpt_path ./ckpts/swin_fulldata.ckpt --output_path Haze --size 512 512 --resize_size 512