CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Rain/ORD.json --model_name nafnet --ckpt_path ./ckpts/nafnet_fulldata_res.ckpt --output_path Rain --size 512 512 --resize_size 512

CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Rain/Rain13K.json --model_name nafnet --ckpt_path ./ckpts/nafnet_fulldata_res.ckpt --output_path Rain --size 512 512 --resize_size 512

CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Rain/Rain1400.json --model_name nafnet --ckpt_path ./ckpts/nafnet_fulldata_res.ckpt --output_path Rain --size 512 512 --resize_size 512

CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Rain/RainDS_synthetic.json --model_name nafnet --ckpt_path ./ckpts/nafnet_fulldata_res.ckpt --output_path Rain --size 512 512 --resize_size 512

CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Rain/RainDS.json --model_name nafnet --ckpt_path ./ckpts/nafnet_fulldata_res.ckpt --output_path Rain --size 512 512 --resize_size 512

CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Rain/RealRain1k.json --model_name nafnet --ckpt_path ./ckpts/nafnet_fulldata_res.ckpt --output_path Rain --size 512 512 --resize_size 512

CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Rain/SPAC.json --model_name nafnet --ckpt_path ./ckpts/nafnet_fulldata_res.ckpt --output_path Rain --size 512 512 --resize_size 512

CUDA_VISIBLE_DEVICES=1 python test.py --json_path testjsons/Rain/LHP_test.json --model_name nafnet --ckpt_path ./ckpts/nafnet_fulldata_res.ckpt --output_path Rain --size 512 512 --resize_size 512 
