from math import log2
import config_SOTA as cfg
from cim_c200_backend.cim_test_tool import CIM_Test
from cim_qn_train.cim_toolchain import *
from cim_weight_mapper.weight_process import *
from model_small import Discriminator, Generator
from utils import *
from torch.utils.data import Dataset, DataLoader, Subset

# NOTE:
"""
This script is based on the code used to obtain the hardware measurement data 
for 'Conventional Heterogeneous ACIM' shown in Figure 4d. For clarity and ease 
of review, the open-source version has been streamlined. 

Since the actual computation requires our ACIM hardware computing platform, it cannot 
be fully reproduced here (the relevant lines are commented out but retained for 
reference). The provided version includes an ACIM dataflow simulator that can 
be used for execution and testing.
"""


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


def get_loader(image_size, num_imgs):
    transform_list = [
        transforms.Resize((image_size, image_size))
    ]

    transform_list.extend([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.Normalize(
            [0.5 for _ in range(cfg.CHANNELS_IMG)],
            [0.5 for _ in range(cfg.CHANNELS_IMG)],
        ),
    ])

    transform = transforms.Compose(transform_list)

    batch_size = cfg.batch_size_end_to_end
    dataset = datasets.ImageFolder(root = cfg.DATASET, transform = transform)

    # Use Subset to limit the dataset to the first num_imgs samples
    subset_indices = list(range(min(num_imgs, len(dataset))))
    subset_dataset = Subset(dataset, subset_indices)

    loader = DataLoader(
        subset_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = cfg.NUM_WORKERS,
        pin_memory = True,
    )
    return loader, subset_dataset


class GAN_QN(CIMToolChain):
    def train_fn(self,
                 critic,
                 gen,
                 loader,
                 step,
                 alpha,
                 opt_critic,
                 opt_gen,
                 ):
        loop = tqdm(loader, leave = True)
        loss_gen_epoch = 0
        loss_critic_epoch = 0

        for batch_idx, (real, _) in enumerate(loop):
            real = real.to(cfg.DEVICE)
            cur_batch_size = real.shape[0]

            noise = torch.randn(cur_batch_size, cfg.Z_DIM, 1, 1).to(cfg.DEVICE)
            fake = gen(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha, step, device = cfg.DEVICE)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + cfg.LAMBDA_GP * gp
                    + (0.001 * torch.mean(critic_real ** 2))
            )
            loss_critic_epoch += loss_critic.cpu().item()

            opt_critic.zero_grad()
            loss_critic.backward()
            opt_critic.step()

            gen_fake = critic(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)
            loss_gen_epoch += loss_gen.cpu().item()

            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            alpha = 1.0

            lr_current = opt_gen.param_groups[0]['lr']
            loop.set_postfix(gp = gp.item(),
                             loss_critic = loss_critic.item(),
                             loss_gen = loss_gen.item(),
                             LR = lr_current)

        return alpha

    def train_model(self, gen, critic, img_size):
        gen.train()
        critic.train()

        step = int(log2(img_size / 4))
        for num_epochs in cfg.epoch_end_to_end:

            alpha = cfg.alpha_start
            print(f"Current image size: {4 * 2 ** step}")

            opt_gen = optim.Adam(gen.parameters(), lr = cfg.lr_end_to_end, betas = (0.0, 0.99))
            opt_critic = optim.Adam(critic.parameters(), lr = cfg.lr_end_to_end, betas = (0.0, 0.99))

            gamma = cfg.lr_end_to_end_factor ** (1 / max((num_epochs - 1), 1))
            scheduler_gen = torch.optim.lr_scheduler.StepLR(opt_gen, step_size = 1, gamma = gamma)
            scheduler_critic = torch.optim.lr_scheduler.StepLR(opt_critic, step_size = 1, gamma = gamma)

            loader, _ = get_loader(4 * 2 ** step,
                                   num_imgs = cfg.end_to_end_train_imgs)
            for epoch in range(num_epochs):
                print(f"Epoch [{epoch + 1}/{num_epochs}]")
                lr_last = opt_gen.param_groups[0]['lr']

                alpha = self.train_fn(
                    critic,
                    gen,
                    loader,
                    step,
                    alpha,
                    opt_critic,
                    opt_gen,
                )

                scheduler_gen.step()
                scheduler_critic.step()

                lr_gen = opt_gen.param_groups[0]['lr']
                print(f'alpha = {alpha}')

                if lr_gen < lr_last:
                    print(f'\n')
                    print(f'---------------------------------------')
                    print(f'LR changed from {lr_last:.3g} to {lr_gen:.3g}')
                    print(f'---------------------------------------')
                    print(f'\n')

                # ====================================== #
                # After each epoch, check the FID at the set frequency and save the weights
                # ====================================== #
                if (epoch + 1) % 2 == 0:
                    FID_img_folder = f'{cfg.main_folder}/FID_imgs/end_to_end_epoch_{epoch}'
                    generate_examples_new(self.model, steps = 5, alpha = 1.0, n = cfg.FID_images,
                                          folder = FID_img_folder,
                                          device = cfg.DEVICE,
                                          noise_dim = 128,
                                          seed = 0,
                                          noise_in = None, add_str = '')

                    filename_gen = f'{cfg.pth_path}/generator_end_to_end_epoch_{epoch}.pth'
                    filename_critic = f'{cfg.pth_path}/critic_end_to_end_epoch_{epoch}.pth'
                    save_model(gen, filename_gen)
                    save_model(critic, filename_critic)

            step += 1  # progress to the next img size

        epoch = cfg.epoch_end_to_end[0]
        FID_img_folder = f'{cfg.main_folder}/FID_imgs/end_to_end_epoch_{epoch}'
        generate_examples_new(self.model, steps = 5, alpha = 1.0, n = cfg.FID_images,
                              folder = FID_img_folder,
                              device = cfg.DEVICE,
                              noise_dim = 128,
                              seed = 0,
                              noise_in = None, add_str = '')
        filename_gen = f'{cfg.pth_path}/generator_end_to_end_epoch_{epoch}.pth'
        filename_critic = f'{cfg.pth_path}/critic_end_to_end_epoch_{epoch}.pth'
        save_model(gen, filename_gen)
        save_model(critic, filename_critic)

    def train_enhance_layer_w_teacher(self,
                                      teacher_model,
                                      path,
                                      epoch = 1,
                                      mode = f'each_layer'):

        training_dataset = get_dataloader(batch_size = cfg.batch_size_self_adaptive,
                                          num_samples = cfg.self_adaptive_train_imgs,
                                          z_dim = cfg.Z_DIM,
                                          device = cfg.DEVICE)

        learning_rate = cfg.lr_self_adaptive
        enhance_parameters = []
        enhance_parameters_name = []
        for name, module in self.model.named_modules():
            if type(module) in reg_dict.nn_layers:
                for p_name, param in module.named_parameters():
                    if param.requires_grad:
                        enhance_parameters.append(param)
                        enhance_parameters_name.append(f'{name}-{p_name}')
        print(f'\n')
        print(f'-----------------------------------')
        print(f'Train Parameters with Teacher Model')
        print(f'-----------------------------------')
        for p_name in enhance_parameters_name:
            print(p_name)

        optimizer = optim.Adam(enhance_parameters, lr = learning_rate)
        print(f'Learning Rate for Fast Self-Adaption Train: {learning_rate}')
        loss_func = nn.MSELoss()

        gamma = cfg.lr_self_adaptive_factor ** (1 / max((cfg.epoch_self_adaptive - 1), 1))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = gamma)
        epoch_loss_list = []
        batch_loss_list = []

        for e in range(epoch):

            loop = tqdm(training_dataset, leave = True)
            epoch_loss = 0
            lr_current = optimizer.param_groups[0]['lr']
            for batch_idx, training_data in enumerate(loop):
                optimizer.zero_grad()

                training_data = training_data.to(self.device)

                teacher_output, teacher_output_dict = self.forward_with_hooks_layer_flag(teacher_model, training_data,
                                                                                         layer_flag = reg_dict.digital_compute_layers)
                student_output, student_output_dict = self.forward_with_hooks_layer_flag(self.model, training_data,
                                                                                         layer_flag = reg_dict.digital_compute_layers)

                total_loss = torch.tensor(0.0, device = training_data.device)

                if mode == f'each_layer':
                    for layer_name, feature_map_t in teacher_output_dict.items():
                        feature_map_s = student_output_dict[layer_name]
                        loss = loss_func(feature_map_s, feature_map_t)
                        total_loss += loss

                elif mode == f'final_layer':
                    total_loss = loss_func(student_output, teacher_output)

                epoch_loss += total_loss.item() / len(training_data)
                batch_loss_list.append(total_loss.item() / len(training_data))

                total_loss.backward()
                optimizer.step()

                loop.set_description(f"Epoch {e + 1}/{epoch} Batch {batch_idx + 1}/{len(training_dataset)}, Loss: {total_loss.item():.4f}")
                save_to_json(batch_loss_list, filename = f'{path}/self_adaptive_train_{mode}_batch_loss_list.json')


            scheduler.step()
            lr_new = optimizer.param_groups[0]['lr']

            if lr_new < lr_current:
                print(f'\n')
                print(f'---------------------------------------')
                print(f'LR changed from {lr_current:.3g} to {lr_new:.3g}')
                print(f'---------------------------------------')
                print(f'\n')
            epoch_loss_list.append(epoch_loss)
            save_to_json(epoch_loss_list, filename = f'{path}/self_adaptive_train_{mode}_epoch_loss_list.json')

            FID_img_folder = f'{cfg.main_folder}/FID_imgs/self_adaption_epoch_{e}'
            generate_examples_new(self.model, steps = 5, alpha = 1.0, n = cfg.FID_images,
                                  folder = FID_img_folder,
                                  device = cfg.DEVICE,
                                  noise_dim = 128,
                                  seed = 0,
                                  noise_in = None, add_str = '')

            filename_gen = f'{cfg.pth_path}/generator_self_adaption_epoch_{e}.pth'
            save_model(gen, filename_gen)

        print(f'\n')
        print(f'Fast Self-Adaption Training Finished!')
        print(f'-----------------------------------')
        self.compare_model_weights(teacher_model, self.model)
        print(f'-----------------------------------')
        print(f'\n')


if __name__ == "__main__":
    # =========================== #
    # Initialize GAN
    # =========================== #
    start_img_size = 128
    step = int(log2(start_img_size / 4))

    gen = Generator(z_dim = cfg.Z_DIM,
                    in_channels = cfg.IN_CHANNELS,
                    img_channels = cfg.CHANNELS_IMG,
                    ).to(cfg.DEVICE)

    critic = Discriminator(
        cfg.Z_DIM, cfg.IN_CHANNELS, img_channels = cfg.CHANNELS_IMG
    ).to(cfg.DEVICE)

    gan = GAN_QN(model = gen, name = cfg.model_name, device = cfg.DEVICE)

    # =========================== #
    # Convert target layers to ACIM hardware simulator runtime
    # =========================== #
    # Layers that are mapped to ACIM cores
    # For conventional SOTA works, layer-wise hybrid mapping is used in this implementation as a benchmark baseline
    assign_ACIM_layers = [
        # 'initial.3.conv',             # -> digital core
        'prog_blocks.0.conv1.conv',     # -> ACIM core
        # 'prog_blocks.0.conv2.conv',   # -> digital core
        'prog_blocks.1.conv1.conv',     # -> ACIM core
        # 'prog_blocks.1.conv2.conv',   # -> digital core
        'prog_blocks.2.conv1.conv',     # -> ACIM core
        # 'prog_blocks.2.conv2.conv',   # -> digital core
        'prog_blocks.3.conv1.conv',     # -> ACIM core
        # 'prog_blocks.3.conv2.conv',   # -> digital core
        'prog_blocks.4.conv1.conv',     # -> ACIM core
        # 'prog_blocks.4.conv2.conv',   # -> digital core
    ]


    gan.convert_to_layers(convert_layer_type_list = reg_dict.op_layers,
                          assign_layers = assign_ACIM_layers,
                          tar_layer_type = 'layers_qn_lsq',
                          noise_scale = 0,
                          input_bit = 8,
                          output_bit = 8,
                          weight_bit = 4, )

    gan.convert_to_layers(convert_layer_type_list = reg_dict.custom_layers,
                          tar_layer_type = 'layers_qn_lsq_adda_cim',
                          noise_scale = 0,
                          input_bit = 8,
                          output_bit = 8,
                          weight_bit = 4,
                          adc_bit = 4,
                          dac_bit = 2,
                          adc_gain_1_scale = 1 / 63,
                          adc_gain_range = [1, 255],
                          gain_noise_scale = 0,
                          offset_noise_scale = 0,
                          adc_adjust_mode = 'gain'
                          )

    # =========================== #
    # Load pre-trained model
    # =========================== #
    gen_model_name = fr'pretrained_model/generator_w_4b_o_8b_SOTA.pth'
    critic_model_name = fr'pretrained_model/critic_w_4b_o_8b_SOTA.pth'
    gan.load_model(gen_model_name, strict = True)
    load_model(critic, critic_model_name, device = cfg.DEVICE)
    gan.set_device(cfg.DEVICE, device_ids = cfg.device_ids)
    # Export ONNX
    # input_data = torch.randn(1, cfg.Z_DIM, 1, 1).to('cuda')
    # gan.export_onnx(input_data, onnx_path = f'GAN_SOTA')
    # =========================== #
    # Mapping ACIM weights
    # =========================== #
    model_weight_mapping_info = map_weight_for_model(gan.model,
                                                     draw_folder = 'Array_Mapping_Info_SOTA',
                                                     array_size = [576, 128],
                                                     weight_block_size = [576, 128])

    # =========================== #
    # Generated Ideal Images
    # =========================== #
    FID_img_folder = f'{cfg.main_folder}/FID_imgs/ideal'
    generate_examples_new(gan.model, steps = 5, alpha = 1.0, n = cfg.FID_images,
                          folder = FID_img_folder,
                          device = cfg.DEVICE,
                          noise_dim = 128,
                          seed = 0,
                          noise_in = None, add_str = '')

    # =========================== #
    # Initialize Teacher Model
    # =========================== #
    # Make a copy of the pre-trained model as the teacher model for fast self-adaption online learning
    teacher_model = copy.deepcopy(gan.model).eval()

    # Add noise to the ACIM simulator
    gan.update_layer_parameter(
        update_layer_type_list = ['layers_qn_lsq_adda_cim'],
        noise_scale = 0.05,
        gain_noise_scale = 0.05,
        offset_noise_scale = 0.05,
    )

    # =========================== #
    # Convert ACIM layers to Hardware Runtime
    # =========================== #
    # Convert ACIM simulator layers to ACIM hardware runtime layers
    # Requires a hardware platform and SDK for neural network computation
    # Uncomment the following two lines to run on actual hardware

    # gan.convert_to_layers(convert_layer_type_list = reg_dict.custom_layers,
    #                       tar_layer_type = 'layers_lsq_144k_FPGA_expansion',
    #                       noise_scale = 0,
    #                       input_bit = 8,
    #                       output_bit = 8,
    #                       weight_bit = 4,
    #                       adc_bit = 4,
    #                       dac_bit = 2,
    #                       adc_gain_1_scale = 1 / 63,
    #                       adc_gain_range = [1, 255],
    #                       adc_adjust_mode = 'gain'
    #                       )
    # map_weight_for_model(gan.model,
    #                      array_size = [576, 128],
    #                      weight_block_size = [576, 128])

    # =========================== #
    # Program weights to ACIM chips
    # =========================== #
    # The following two lines deploy ACIM weights to a real ACIM chip, which requires a hardware computing platform to run.
    # If using a simulator, comment out the following two lines.

    # program_tool = CIM_Test(model = gan.model, name = 'GAN_SOTA')
    # program_tool.chip_test_model(save_path = f'{cfg.main_folder}/Chip_test',
    #                              prog_cycle = 10)

    # =========================== #
    # Generated Images Before Online Learning
    # =========================== #
    FID_img_folder = f'{cfg.main_folder}/FID_imgs/untrained'
    generate_examples_new(gan.model, steps = 5, alpha = 1.0, n = cfg.FID_images,
                          folder = FID_img_folder,
                          device = cfg.DEVICE,
                          noise_dim = 128,
                          seed = 0,
                          noise_in = None, add_str = '')

    # =========================== #
    # Online Learning Stage 1: Fast Self-Adaption
    # =========================== #
    gan.freeze_backbone()
    training_dataset = get_dataloader(batch_size = cfg.batch_size_self_adaptive,
                                      num_samples = cfg.self_adaptive_train_imgs,
                                      z_dim = cfg.Z_DIM,
                                      device = cfg.DEVICE)

    for name, module in gan.model.named_modules():
        if type(module) in reg_dict.op_layers:
            module.layer_flag = 'enhance_layer'

    for name, module in teacher_model.named_modules():
        if type(module) in reg_dict.op_layers:
            module.layer_flag = 'enhance_layer'

    # Online Learning Stage 1: Fast Self-Adaption
    gan.train_enhance_layer_w_teacher(teacher_model = teacher_model,
                                      epoch = cfg.epoch_self_adaptive,
                                      path = f'{cfg.main_folder}/self_adaptive_train',
                                      mode = 'each_layer'
                                      )

    print(f'Finished Fast Self-Adaption !!')
    # =========================== #
    # Online Learning Stage 2: End-to-end training
    # =========================== #
    start_img_size = 128
    gan.train_model(gen = gan.model,
                    critic = critic,
                    img_size = start_img_size
                    )
    print(f'Finished End-to-end Training !!')
