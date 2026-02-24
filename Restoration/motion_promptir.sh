CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Motion\ blur/GoPro.json --model_name promptir --ckpt_path ./ckpts/promptir_fulldata_res.ckpt --output_path Motion_blur --size 512 512

CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Motion\ blur/HIDE.json --model_name promptir --ckpt_path ./ckpts/promptir_fulldata_res.ckpt --output_path Motion_blur --size 512 512

CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Motion\ blur/RealBlur.json --model_name promptir --ckpt_path ./ckpts/promptir_fulldata_res.ckpt --output_path Motion_blur --size 512 512

CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Motion\ blur/REDS.json --model_name promptir --ckpt_path ./ckpts/promptir_fulldata_res.ckpt --output_path Motion_blur --size 512 512
