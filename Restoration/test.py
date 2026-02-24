import torch.optim as optim

from utils.image_utils import unnormalize
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from utils.dataloader import *
from net.promptir import *

from utils.image_io import *
from utils.val_utils import *

import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        if opt.model_name[0] == 'swin':
            from net.swin import build_swin
            self.net = build_swin(opt)
        elif opt.model_name[0] == 'promptir':
            from net.promptir import PromptIR
            self.net = PromptIR(decoder=True)
        elif opt.model_name[0] == 'nafnet':
            from net.nafnet import NAFNet
            self.net = NAFNet()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # Ignore
        ([clean_name, de_id, dataset, category], degrad_patch, clean_patch) = batch

        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        return loss

    def validation_step(self, batch, batch_idx):
        # Ignore
        ([clean_name, de_id, dataset, category], degrad_patch, clean_patch) = batch

    def configure_optimizers(self):
        # Ignore
        optimizer = optim.AdamW(self.parameters(), lr=opt.lr, eps=1e-8, betas=(0.9, 0.999), weight_decay=0.05)
        warmup_iters = self.num_iters * opt.warmup_epochs
        max_iters = self.num_iters * opt.epochs
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=warmup_iters,
                                                  max_epochs=max_iters, eta_min=1e-5)

        return [optimizer], [scheduler]


def test_ds(net, testloader):
    # Initialize metrics
    psnr = AverageMeter()
    ssim = AverageMeter()
    lpips_scores = AverageMeter()

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').cuda()
    fid_metric = FrechetInceptionDistance(feature=2048).cuda()
    lpips_metric.eval()
    fid_metric.eval()
    net.eval()

    stop_count = 500  # Max samples to run
    with torch.no_grad():
        count = 0
        for ([clean_name, degrad_name, dataset, category], degrad_patch, clean_patch, degst, text) in tqdm(testloader):
            if count > stop_count:
                break
            write_dir = os.path.join(opt.model_name[0], opt.output_path, category[0], dataset[0])
            os.makedirs(write_dir, exist_ok=True)

            if dataset[0] == 'SICE':
                deg_name = degrad_name[0].split('/')[-1]

                if int(deg_name.split('.')[0]) > 3:
                    continue

            count += 1

            if dataset[0] == 'SICE':
                clean_name = degrad_name[0].split('/')[-2] + '_' + degrad_name[0].split('/')[-1]
            elif dataset[0] == 'REVIDE':
                clean_name = clean_name[0].split('/')[-2] + '_' + clean_name[0].split('/')[-1]
            elif dataset[0] == 'GoPro':
                clean_name = clean_name[0].split('/')[-3] + '_' + clean_name[0].split('/')[-1]
            else:
                clean_name = degrad_name[0].split('/')[-1]

            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            restored = net(degrad_patch)

            if opt.imagenet_norm:
                restored, clean_patch, degrad_patch = unnormalize([restored, clean_patch, degrad_patch])

            # Metrics
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            restored = restored.clamp(0, 1)
            clean_patch = clean_patch.clamp(0, 1)

            # Prepare images for LPIPS (range [-1,1])
            restored_lpips = 2 * restored - 1
            clean_patch_lpips = 2 * clean_patch - 1

            # Prepare images for FID (range [0,255], uint8)
            restored_fid = restored
            clean_patch_fid = clean_patch
            restored_fid_uint8 = restored_fid.mul(255).byte()
            clean_patch_fid_uint8 = clean_patch_fid.mul(255).byte()

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            lpips_value = lpips_metric(restored_lpips, clean_patch_lpips)
            lpips_scores.update(lpips_value.item(), N)

            fid_metric.update(restored_fid_uint8, real=False)
            fid_metric.update(clean_patch_fid_uint8, real=True)

            save_image_tensor(restored, os.path.join(write_dir, clean_name.replace('.png', '.jpg')))  # Saves space

        # Compute FID after processing all images
        fid_value = fid_metric.compute().item()

        print(
            f"PSNR, SSIM, LPIPS, FID: {psnr.avg:.2f}/{ssim.avg:.3f}/"
            f"{lpips_scores.avg:.4f}/{fid_value:.2f}"
        )


def main():
    print("Options")
    print(opt)

    testloader = create_datasets(opt)

    model = Model().load_from_checkpoint(opt.ckpt_path, strict=False).cuda()

    test_ds(model, testloader)


if __name__ == '__main__':
    main()
