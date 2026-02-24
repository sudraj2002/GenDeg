import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=120, help='maximum number of epochs to train the total model.')
parser.add_argument('--warmup_epochs', type=int, default=10, help='maximum number of epochs to train the total model.')
parser.add_argument('--val_every', type=int, default=10, help='validation every n epochs')
parser.add_argument('--batch_size', type=int,default=8,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=[],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--val_de_type', nargs='+', default=[],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--model_name', nargs='+', default=['promptir', 'swin', 'swin_vis', 'swin_txt', 'swin_1x1'],
                    help='which type of degradations is training and testing for.')


parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--resize_size', type=int, default=224, help='patchsize of input.')
parser.add_argument('--resize', action='store_true', default=False, help='Resume or not')
parser.add_argument('--imagenet_norm', action='store_true', default=False, help='Resume or not')
parser.add_argument('--crop', action='store_true', default=False, help='Resume or not')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')
parser.add_argument('--num_val', type=int, default=16, help='number of workers.')
parser.add_argument('--size', nargs='+',
                        default=None,
                        help='image size')

# path
parser.add_argument('--dir', type=str, default='/data/wriva_all/',  help='where clean images of denoising saves.')
parser.add_argument('--text', type=str, default=None,  help='where clean images of denoising saves.')
parser.add_argument('--json_path', type=str, default='/data/wriva_all/',  help='where clean images of denoising saves.')
parser.add_argument('--image', type=str, default=None,  help='where clean images of denoising saves.')
parser.add_argument('--image_clean', type=str, default=None,  help='where clean images of denoising saves.')
parser.add_argument('--train_file', type=str, default='data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--val_file', type=str, default='data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--server', action='store_true', default=False, help='Resume or not')
parser.add_argument('--allow_others', action='store_true', default=False, help='Resume or not')

parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/", help='checkpoint save path')
parser.add_argument('--resume', action='store_true', default=False, help='Resume or not')
parser.add_argument('--combined', action='store_true', default=False, help='Resume or not')
parser.add_argument("--wblogger",type=str,default="GenIR",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="train_ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default= 2,help = "Number of GPUs to use for training")

options = parser.parse_args()

