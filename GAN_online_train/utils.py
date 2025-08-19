import time
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def gradient_penalty(critic, real, fake, alpha, train_step, device = "cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs = interpolated_images,
        outputs = mixed_scores,
        grad_outputs = torch.ones_like(mixed_scores),
        create_graph = True,
        retain_graph = True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim = 1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_model(model, PATH):
    # 获取目录路径
    dir_path = os.path.dirname(PATH)

    # 检查目录是否存在，如果不存在，则创建它
    if not os.path.exists(dir_path) and dir_path != "":
        os.makedirs(dir_path)
        print('Directory created:', dir_path)

    # 如果是 DDP 包装，取出原始模型
    if isinstance(model, DDP):
        model_to_save = model.module
    else:
        model_to_save = model

    # 保存状态字典
    torch.save(model_to_save.state_dict(), PATH)
    print('Model saved:', PATH)


def remove_module_prefix(state_dict):
    """移除所有键中的'module.'前缀"""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith('module.') else key  # 去除'module.'前缀
        new_state_dict[new_key] = value
    return new_state_dict


def load_model(model, PATH, device):
    checkpoint = torch.load(PATH, map_location = device)
    checkpoint = remove_module_prefix(checkpoint)
    model.load_state_dict(checkpoint, strict = True)
    print('model loaded !')


def generate_examples_new(model, steps, alpha = 1.0, n = 100,
                          batch_size = 1,
                          folder = 'generated_images',
                          device = 'cuda', noise_dim = 128,
                          seed = None,
                          save_noise = False,
                          noise_in = None, add_str = ''):
    if seed is not None:
        # 如果使用CUDA
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
        torch.manual_seed(seed)
    model.eval()
    if not os.path.exists(folder):  # 检查文件夹是否存在
        os.makedirs(folder)  # 如果不存在，创建文件夹

    batches = (n + batch_size - 1) // batch_size  # 计算总批次数，向上取整
    remaining = n  # 记录剩余需要生成的图片数
    t = time.time()
    for i in tqdm(range(batches), desc = "Generating image batches", unit = "batch"):
        current_batch_size = min(batch_size, remaining)  # 确保最后一批次不会超出 n

        with torch.no_grad():
            noise = torch.randn(current_batch_size, noise_dim, 1, 1, device = device)

            img = model(noise, alpha, steps)

            for j in range(current_batch_size):
                save_image(
                    img[j] * 0.5 + 0.5,
                    os.path.join(folder, f"img_{i * batch_size + j}_s_{steps}_{add_str}.png")
                )

            if save_noise:
                torch.save(noise, os.path.join(folder, f"img_batch_{i}_s_{steps}_{add_str}.pth"))

        remaining -= current_batch_size  # 更新剩余需要生成的图片数
    model.train()
    t = time.time() - t
    print(f'Time for generating {n} images: {t:.3f}s')
    return img


def generate_examples_fix(gen, noise_in, name, steps = 5, alpha = 1.0, n = 1,
                          folder = 'compare_examples'):
    gen.eval()
    if not os.path.exists(folder):  # 检查文件夹是否存在
        os.makedirs(folder)  # 如果不存在，创建文件夹
    for i in range(n):
        print(f'generating image {i + 1}/{n}')
        with torch.no_grad():
            img = gen(noise_in, alpha, steps)
            save_image(img * 0.5 + 0.5, os.path.join(folder, f"{name}.png"))
    gen.train()
    return img


class NoiseDataset(Dataset):
    def __init__(self, num_samples, z_dim, device):
        self.num_samples = num_samples
        self.z_dim = z_dim
        self.device = device

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        noise = torch.randn(self.z_dim, 1, 1).to(self.device)
        return noise


def get_dataloader(batch_size, num_samples, z_dim, device):
    dataset = NoiseDataset(num_samples, z_dim, device)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    return dataloader
