'''
@author: Wang Ze
'''

import matplotlib.pyplot as plt
import numpy as np

from .cim_utils import *


def ADC_auto_adjust(chip_idx,
                    input_matrix,
                    addr,
                    target_percent,
                    dac_bits = 2,
                    weight = None,
                    verbose = 1):
    def print_verbose(message):
        if verbose:
            print(message)

    def calculate_adc_gain_change():
        nonlocal adc_gain, too_large_flag, too_small_flag

        if max_per <= low_lim:
            too_large_flag = False
            if too_small_flag:
                adc_gain *= 2
            else:
                adc_gain += step / 2
        elif max_per >= high_lim:
            too_small_flag = False
            if too_large_flag:
                adc_gain /= 2
            else:
                adc_gain -= step / 2

    def is_search_failed():
        return count == 0 or 0 < step <= 0.5 or adc_gain_pre >= adc_gain_levels_max

    def log_midpoint(low_lim, high_lim):

        midpoint_log = (math.log10(low_lim) + math.log10(high_lim)) / 2

        midpoint = 10 ** midpoint_log
        return midpoint

    low_lim = target_percent[0]
    high_lim = target_percent[1]
    mid_point = log_midpoint(low_lim, high_lim)

    max_per = 1
    adc_gain_list = []
    overshoot_percent_list = []

    adc_gain = 1
    adc_gain_levels = np.arange(1, 64, 1)
    adc_gain_levels_max = adc_gain_levels.max()

    step = adc_gain

    count = 30

    too_small_flag = 1
    too_large_flag = 1

    while not (low_lim <= max_per <= high_lim):
        adc_gain = max(1, min(adc_gain, adc_gain_levels_max))
        adc_gain_pre = adc_gain
        adc_gain = round(adc_gain)
        print_verbose(f'--------------------------\nTrying adc_gain @ {adc_gain}')

        if weight is None:
            chip_weight = np.load(f'chip_idx_0.npy')
            weight = chip_weight[addr[0]:addr[0] + addr[2], addr[1]:addr[1] + addr[3]]

        mvm_result, ADC_output, ADC_scale = mvm_calculate(chip_idx,
                                                          weight = weight,
                                                          input_matrix = input_matrix,
                                                          addr = addr,
                                                          noise_scale = 0,
                                                          dac_bits = 2,
                                                          use_simulator = 1,
                                                          it_time = adc_gain)

        mvm_expected = np.dot(input_matrix.transpose(1, 0), weight)
        max_per = len(ADC_output[ADC_output == 7]) / np.prod(ADC_output.shape)
        adc_gain_list.append(adc_gain)
        overshoot_percent_list.append(max_per)
        print_verbose(f'ADC overshoot percent = {max_per * 100:.2f}%')
        if verbose:
            scatter_plt(mvm_result / ADC_scale, mvm_expected,
                        title = f'Overshoot Percent = {max_per * 100:.2f}%',
                        show_fig = 1, save_fig = 0, range_abs = None)

        calculate_adc_gain_change()
        step = abs(adc_gain - adc_gain_pre)
        print_verbose(f'step = {step}')
        count -= 1

        if is_search_failed():
            distance = (np.array(overshoot_percent_list) - mid_point) ** 2
            idx = np.argmin(distance)
            return adc_gain_list[idx]

    return adc_gain
