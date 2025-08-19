import torch
import os
# ===================== #
# Computing Platform
# ===================== #

num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")
if num_gpus > 1:
    run_env = 'a100'
else:
    run_env = 'pc'
print(f'Running on {run_env}')

# ============================= #
# Adjustable Parameters
# ============================= #
seed = 5
FID_images = 100
model_name = f'GAN_online_learning_DORA'

# ============================= #
# Device Selection
# ============================= #
if run_env == 'a100':
    DATASET = fr'../../dataset/celeb_dataset_LR_128'
    DEVICE = 'cuda:6'
    num_workers = 4
else:
    DATASET = fr'../dataset'
    DEVICE = 'cuda' if torch.cuda.is_available() else torch.device('cpu')
    device_ids = None
    num_workers = 0
device_ids = None

# ============================= #
# General NN Parameters
# ============================= #
CHANNELS_IMG = 3
Z_DIM = 128
IN_CHANNELS = 128
LAMBDA_GP = 10
NUM_WORKERS = 0
alpha_start = 1

# ======================= #
# Save Path
# ======================= #
main_folder = f'{model_name}'

pth_path = f'{main_folder}/trained_pth'
img_path = f'{main_folder}/training_img'

# ======================= #
# Fast self-adaption params
# ======================= #
self_adaptive_train_imgs = 100
epoch_self_adaptive = 4
lr_self_adaptive = 1e-2
lr_self_adaptive_factor = 1e-1
batch_size_self_adaptive = 2

# ======================= #
# End-to-end training params
# ======================= #
end_to_end_train_imgs = 100
epoch_end_to_end = [3]
lr_end_to_end = 1e-4
lr_end_to_end_factor = 1e-1
batch_size_end_to_end = 4