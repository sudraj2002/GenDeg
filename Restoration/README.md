# Restoration models

Instruction for testing restoration models trained using GenDS dataset.

---

## Getting Started

### Requirements

Set up the environment:

```bash
conda create -n test python=3.8 -y
pip install -r requirements.txt
```

---

## Data & Model Download

- Download the test dataset (`Test/`) from **[this link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/sambasa2_jh_edu/IQB5Fmetot5iRKPDmQGgXAOgAcbcDK7rvxwmgVY35A-pTRw?e=SO4DUb)** and place it under `Restoration/`.
- Download metadata files (`testjsons/`) from **[this link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/sambasa2_jh_edu/IQD52Lxg6bK8RrPoX3_hYVmmAV8qCUe9NJXV9nfZkbl4fSA?e=YbEt2C)** and place it under `Restoration/`.

- Download pre-trained weights of the models from **[this link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/sambasa2_jh_edu/IQBsK-SxncexQpxB1FDF6vuWAdtbfpEnAansay4QkvntHzY?e=IIS1jr)** and place them in `ckpts/`. 

For comparisons, we recommend comparing with the NAFNet model as it performs best. For models other than PromptIR, Swin and NAFNet, use their respective
codebases for testing.

---

## Inference

Based on desired degradation and model, simply run

```bash
bash haze_nafnet.sh # E.g. this is for dehazing with NAFNet trained on GenDS.
```

## Acknowledgments

This codebase is built upon [PromptIR](https://github.com/va1shn9v/PromptIR). We thank the authors for sharing their code!

---
