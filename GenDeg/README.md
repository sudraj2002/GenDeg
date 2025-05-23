# GenDeg

GenDeg is a controllable diffusion model for generating degraded images. This code can be used for generating large-scale synthetic degradation datasets such as [GenDS](https://huggingface.co/datasets/Sudarshan2002/GenDS).

---

## Getting Started

### Requirements

Set up the environment:

```bash
conda create -f environment.yml
conda activate <env_name>
pip install lmdb
```

Set up NAFNet (for structure correction):

```bash
cd NAFNet
python setup.py develop --no_cuda_ext
```

---

## Data & Model Download

- Download the base restoration dataset (`GenIRData/`) from **[this link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/sambasa2_jh_edu/ERKv9aDcj4BErJ8brU2oDn4BmIIiokwWitpHmBuiPtCfYQ?e=jiJZZg)** and note the absolute path to where you place the dataset.
- The file `data/seeds.json` contains image paths required for training and dataset generation.
- `train_csv.json` provides the list of clean image paths and associated captions generated by [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2).

Download pre-trained weights of the generator (**[this link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/sambasa2_jh_edu/EV8NY5aUDdNKo1SkvU92Fi8B2r53oS7pqlBmoyC5dVbLAQ?e=XhtfQk)**) and structure correction network (**[this link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/sambasa2_jh_edu/EfAIJSa9JfxJvSng_a9Xcp4BTwUmggXBnr9u9Jj_omTnfA?e=HJUQOm)**) and place them under `checkpoints/`.

---

## Inference

### GenDS Dataset Generation

Generate degraded images for a particular degradation:

```bash
python run_parallel_mu_sigma.py \
    --deg_type <haze | rain | snow | motion | low-light | raindrop> \
    --output <output_directory> \
    --replace_dir <path_to_GenIRData> \
    --input data \
    --auto_s
```

You can generate images using your own data as well by creating `seeds.json` and `train_csv.json` in appropriate formats

#### Optional arguments

| Argument      | Description                                |
|---------------|--------------------------------------------|
| `--s`         | Enable structure correction for all images |
| `--resolution`| Image generation resolution                |
| `--batch_size`| Number of images to generate in parallel   |
| `--steps`     | Number of sampling steps                   |
| `--cfg-text`  | Text guidance scale                        |
| `--cfg-image` | Image guidance scale                       |

---

### Single Image Inference

Run generation on a single image:

```bash
python run_single.py \
    --img <path_to_image> \
    --prompt "<your_prompt>" \
    --mu <optional_mu_value> \
    --sigma <optional_sigma_value> \
    --deg_type <if mu/sigma not provided>
```

Other arguments (e.g., `--cfg-text`, `--resolution`) are shared with `run_parallel_mu_sigma.py`.

---

## Training

To train GenDeg:

```bash
bash train.sh --replace_dir <path_to_GenIRData>
```

You can also train GenDeg on your own data by making the appropriate changes in `train_csv.json` and `seeds.json`.

---

## Acknowledgments

This codebase is built upon [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix). We thank the authors for sharing their code!

---
