'''
@author: Wang Ze
'''

from cim_layers.quant_noise_utils import *
import math


def init_adc_gain(module, out_, adc_adjust_mode = 'gain'):
    if out_.abs().max() != 0:
        adc_scale_ideal = module.adc_range / out_.abs().max()
        adc_gain_ideal = adc_scale_ideal / module.adc_gain_1_scale
        adc_gain_ideal = torch.clamp(adc_gain_ideal, min = 0.8 * module.adc_gain_min, max = 1.2 * module.adc_gain_max)

        module.adc_gain.data.copy_(adc_gain_ideal.to(module.adc_gain.device))
        module.adc_scale = get_adc_scale(module, module.adc_gain, adc_adjust_mode)
        print(f'Initialized Adc Gain: {module.adc_gain.data}')


def init_adc_gain_(module, out_):
    if out_.abs().max() != 0:
        adc_scale_ideal = module.adc_range / out_.abs().max()
        adc_gain_ideal = adc_scale_ideal / module.adc_gain_1_scale
        adc_gain_ideal = torch.clamp(adc_gain_ideal, min = 0.8 * module.adc_gain_min, max = 1.2 * module.adc_gain_max)
        print(f'Initialized Adc Gain: {adc_gain_ideal}')
    else:
        return None
    return adc_gain_ideal


def get_adc_scale(module, adc_gain, mode = 'gain'):
    adc_gain = clamp_pass(adc_gain, min = module.adc_gain_min, max = module.adc_gain_max)
    if mode == 'gain':
        adc_gain = round_pass(adc_gain)
    else:
        adc_range = round_pass(1 / adc_gain)
        adc_gain = 1 / adc_range
    adc_scale = adc_gain * module.adc_gain_1_scale
    return adc_scale


def update_adc_gain(module, adc_bit_old, dac_bit_old, weight_bit_old):
    adc_gain_old = module.adc_gain.data.item()
    adc_gain_new = adc_gain_old
    if adc_bit_old != module.adc_bit:
        adc_range_factor = 2 ** (module.adc_bit - adc_bit_old)
        adc_gain_new *= adc_range_factor

    if dac_bit_old != module.dac_bit:
        dac_range_factor = 2 ** (module.dac_bit - dac_bit_old)
        adc_gain_new /= dac_range_factor

    if weight_bit_old != module.weight_bit:
        weight_range_factor = 2 ** (module.weight_bit - weight_bit_old)
        adc_gain_new /= weight_range_factor

    adc_gain_new = torch.tensor(adc_gain_new, device = module.adc_gain.device)
    adc_gain_new = torch.clamp(adc_gain_new, min = 0.8 * module.adc_gain_min, max = 1.2 * module.adc_gain_max)

    module.adc_gain.data = adc_gain_new

    if adc_gain_old != module.adc_gain.data.item():
        print(f'adc_gain changed: {adc_gain_old} -> {adc_gain_new}')


def update_adc_gain_multi(module, adc_bit_old, dac_bit_old, weight_bit_old):
    for key, adc_gain_old in module.adc_gain_dict.items():
        adc_gain_old = adc_gain_old.data.item()
        adc_gain_new = adc_gain_old
        if adc_bit_old != module.adc_bit:
            adc_range_factor = 2 ** (module.adc_bit - adc_bit_old)
            adc_gain_new *= adc_range_factor

        if dac_bit_old != module.dac_bit:
            dac_range_factor = 2 ** (module.dac_bit - dac_bit_old)
            adc_gain_new /= dac_range_factor

        if weight_bit_old != module.weight_bit:
            weight_range_factor = 2 ** (module.weight_bit - weight_bit_old)
            adc_gain_new /= weight_range_factor

        adc_gain_new = torch.tensor(adc_gain_new, device = module.adc_gain.device)
        adc_gain_new = torch.clamp(adc_gain_new, min = 0.8 * module.adc_gain_min, max = 1.2 * module.adc_gain_max)
        module.adc_gain_dict[key].data = adc_gain_new

        if adc_gain_old != module.adc_gain_dict[key].data.item():
            print(f'adc_gain changed: {adc_gain_old} -> {adc_gain_new}')


def gen_adc_noise(module):
    torch.manual_seed(module.seed)
    module.gain_noise = torch.randn(1000, device = module.weight.device) * module.gain_noise_scale

    module.offset_noise = torch.randn(1000, device = module.weight.device) * module.offset_noise_scale


def add_adc_noise(module, out_adc, start_col, col_num):
    if module.gain_noise_scale == 0 and module.offset_noise_scale == 0:
        return out_adc
    gain_noise_ = module.gain_noise[start_col:start_col + col_num]
    offset_noise_ = module.offset_noise[start_col:start_col + col_num]
    out_adc_gn = out_adc * (1 + gain_noise_)
    out_adc_off_n = out_adc_gn + module.adc_range * offset_noise_
    out_adc_off_n = (out_adc_off_n - out_adc).detach() + out_adc
    return out_adc_off_n


def bit_split_tensor(x_q, x_bit, slice_bit):
    assert slice_bit >= 1
    bit_data_list = []
    bit_len = int(math.ceil((x_bit - 1) / slice_bit))
    for b in range(0, x_bit - 1, slice_bit):
        lsb = b
        msb = min(b + slice_bit, x_bit - 1)
        shift_data = floor_pass(x_q / 2 ** lsb)
        residue_data = floor_no_pass(x_q / 2 ** msb) * 2 ** slice_bit
        bit_data = shift_data - residue_data
        bit_data_pass = (bit_data - shift_data / bit_len).detach() + shift_data / bit_len
        bit_data_list.append(bit_data_pass)
    bit_data_list = torch.cat(bit_data_list, 0)
    return bit_data_list


def bit_concat_tensor(bitwise_data, data_bit, slice_bit):
    bit_len = int(math.ceil((data_bit - 1) / slice_bit))
    bitwise_data = torch.chunk(bitwise_data, chunks = bit_len, dim = 0)
    data_sum = torch.zeros_like(bitwise_data[0], device = bitwise_data[0].device)
    for i, bit_data in enumerate(bitwise_data):
        data_sum += bit_data * 2 ** (i * slice_bit)
    return data_sum
