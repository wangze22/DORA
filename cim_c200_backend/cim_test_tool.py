'''
@author: Wang Ze
'''
import time

import cim_layers.register_dict as reg_dict
from c200_sdk.sdk_array_newsystem import SDKArray
from cim_layers.quant_noise_utils import *
from cim_runtime_simple import cim_adc as simple_cim
from cim_toolchain_utils.utils import *
from cim_c200_backend.utils import *


def find_high_std_chips(ret_dict, std_threshold):
    def adjusted_threshold(weight_rows, base_threshold, min_rows = 10, min_factor = 0.75):
        if weight_rows <= min_rows:
            return base_threshold * min_factor
        return base_threshold

    high_std_chips = {}

    for array_idx, weights in ret_dict.items():
        for weight_key, metrics in weights.items():
            weight_rows = metrics['weight_rows']
            threshold = adjusted_threshold(weight_rows, std_threshold)

            if metrics['diff_std_relative'] > threshold:
                if array_idx not in high_std_chips:
                    high_std_chips[array_idx] = []
                high_std_chips[array_idx].append(weight_key)

    return high_std_chips


def calculate_average_std(ret_dict):
    average_std_dict = {}

    for array_idx, weights in ret_dict.items():
        total_std = 0
        count = 0

        for weight_key, metrics in weights.items():
            total_std += metrics['diff_std_relative']
            count += 1

        if count > 0:
            average_std = total_std / count
            average_std_dict[array_idx] = average_std

    return average_std_dict


class CIM_Test():
    def __init__(self, name, model = None):
        self.model = model
        self.name = name

    def get_rram_weight(self, test_array = None, skip_array = None, repeat = 10):
        t_tot = time.time()

        assert self.model is not None
        weight_dict = extract_weight_data(model = self.model)
        weight_dict_rram = copy.deepcopy(weight_dict)
        weight_arrays = weight_dict.keys()
        test_array_list = get_test_array_list(weight_arrays, test_array, skip_array)

        ret_dict = {}

        total_weight_size = calculate_total_weight_size(weight_dict, test_array_list)
        tested_weight_size = 0

        for array_idx, data in weight_dict.items():
            if array_idx in test_array_list:
                pass
            else:
                continue

            t_array = time.time()
            print(f'=========================')
            print(f'Reading Chip {array_idx} / {len(weight_dict) - 1}')
            ret_dict[f'array_{array_idx}'] = {}

            sdk = SDKArray(array_idx)

            for idx, data_dict in data.items():
                t_weight = time.time()
                weight_addr = data_dict['addr']
                weight_data = data_dict['data']

                print(f'    ----------------------')
                print(f'    Reading Weight {idx} / {len(data) - 1}')

                rram_weight = sdk.get_weight_int4(addr = weight_addr)
                rram_weight = np.float32(rram_weight)
                for rr in range(repeat - 1):
                    rram_weight += sdk.get_weight_int4(addr = weight_addr)
                rram_weight /= repeat
                weight_dict_rram[array_idx][idx]['data'] = rram_weight

                t_weight = time.time() - t_weight
                print(f'Time for weight_{idx} = {t_weight:.2f}s')

                tested_weight_size += weight_data.size
                estimated_time = estimate_remaining_time(t_tot, tested_weight_size, total_weight_size)
                print(f'Estimated remaining time: {estimated_time:.2f}s; ({estimated_time / 60:.2f}min)')

            t_array = time.time() - t_array
            print(f'Time for Array {array_idx} = {t_array:.2f}s')
        return weight_dict, weight_dict_rram

    def set_rram_weight(self, test_array = None, skip_array = None, prog_cycle = 80):
        t_tot = time.time()

        assert self.model is not None
        weight_dict = extract_weight_data(model = self.model)
        weight_dict_rram = copy.deepcopy(weight_dict)
        weight_arrays = weight_dict.keys()
        test_array_list = get_test_array_list(weight_arrays, test_array, skip_array)

        ret_dict = {}

        total_weight_size = calculate_total_weight_size(weight_dict, test_array_list)
        tested_weight_size = 0

        for array_idx, data in weight_dict.items():
            if array_idx in test_array_list:
                pass
            else:
                continue

            t_array = time.time()
            print(f'=========================')
            print(f'Programing Chip {array_idx} / {len(weight_dict) - 1}')
            ret_dict[f'array_{array_idx}'] = {}

            sdk = SDKArray(array_idx)

            for idx, data_dict in data.items():
                t_weight = time.time()
                weight_addr = data_dict['addr']
                weight_data = data_dict['data']

                print(f'    ----------------------')
                print(f'    Programing Weight {idx} / {len(data) - 1}')
                sdk.set_weight_int4(weight_data,
                                    addr = weight_addr,
                                    prog_cycle = prog_cycle)

                t_weight = time.time() - t_weight
                print(f'Time for weight_{idx} = {t_weight:.2f}s')

                tested_weight_size += weight_data.size
                estimated_time = estimate_remaining_time(t_tot, tested_weight_size, total_weight_size)
                print(f'Estimated remaining time: {estimated_time:.2f}s; ({estimated_time / 60:.2f}min)')

            t_array = time.time() - t_array
            print(f'Time for Array {array_idx} = {t_array:.2f}s')
        return weight_dict, weight_dict_rram

    def chip_test_manually(self, test_array_list,
                           weight_addr = [0, 0, 128, 128], weight_seed = 1,
                           break_bad_chips = True, col_color = True,
                           plt_fig = False, save_fig = 1,
                           save_path = 'chip_test', bad_std_threshold = 0.07,
                           program_weight = True, prog_cycle = 80):

        t_tot = time.time()
        ret_dict = {}
        tested_array_num = 0
        for array_idx in test_array_list:
            t_array = time.time()
            print(f'=========================')
            print(f'Testing Chip {array_idx} / {len(test_array_list) - 1}')
            ret_dict[f'array_{array_idx}'] = {}

            sdk = SDKArray(array_idx)

            np.random.seed(weight_seed)
            weight_data = np.random.randn(weight_addr[2], weight_addr[3])
            weight_data, _ = data_quant(data_float = weight_data, isint = 1, data_bit = 4)

            if program_weight:
                print(f'    ----------------------')
                print(f'    Programing Weight')
                sdk.set_weight_int4(weight_data,
                                    addr = weight_addr,
                                    prog_cycle = prog_cycle)

            input_rows = weight_addr[2]
            input_cols = 200
            input_matrix = np.random.randn(input_rows, input_cols)
            input_matrix[input_matrix < 0] = 0
            input_matrix_q, input_quant_scale = data_quant(input_matrix,
                                                           data_bit = 8,
                                                           isint = 1)

            adc_gain_best = simple_cim.ADC_auto_adjust(chip_idx = array_idx,
                                                       input_matrix = input_matrix_q,
                                                       addr = weight_addr,
                                                       weight = weight_data,
                                                       target_percent = [0.002, 0.001],
                                                       verbose = 0)

            print(f'    ----------------------')
            print(f'    Testing MVM')
            [mvm_result,
             ADC_output,
             ADC_scale] = simple_cim.mvm_calculate(array_idx,
                                                   weight = weight_data,
                                                   input_matrix = input_matrix_q,
                                                   use_simulator = 0,
                                                   addr = weight_addr,
                                                   it_time = adc_gain_best)

            mvm_result /= ADC_scale

            mvm_expected_int = np.dot(input_matrix_q.transpose(1, 0), weight_data)

            diff = mvm_expected_int - mvm_result
            data_range = mvm_expected_int.flatten().max() - mvm_expected_int.flatten().min()
            relative_std = diff.std() / data_range

            ret_dict[f'array_{array_idx}'][f'weight_1'] = {}
            ret_dict[f'array_{array_idx}'][f'weight_1']['diff_mean'] = diff.mean()
            ret_dict[f'array_{array_idx}'][f'weight_1']['diff_std_relative'] = relative_std
            ret_dict[f'array_{array_idx}'][f'weight_1']['weight_rows'] = weight_addr[2]
            ret_dict[f'array_{array_idx}'][f'weight_1']['mvm_expected'] = mvm_expected_int
            ret_dict[f'array_{array_idx}'][f'weight_1']['mvm_result'] = mvm_result

            print(f'array_{array_idx}||diff_std_relative = {relative_std}')

            if relative_std >= bad_std_threshold and break_bad_chips:
                print(f'Array {array_idx} Failed MVM Test, diff_std_relative = {relative_std}')

            plot_weight_test_results(ret_dict, array_idx, plt_fig, save_fig = save_fig, col_color = col_color,
                                     save_path = save_path)
            tested_array_num += 1
            t_array = time.time() - t_array
            t_tot_passed = time.time() - t_tot
            t_avg = t_tot_passed / tested_array_num
            print(f'Time for Array {array_idx} = {t_array:.2f}s')
            estimated_time = t_avg * (len(test_array_list) - tested_array_num)
            print(f'Estimated remaining time: {estimated_time:.2f}s; ({estimated_time / 60:.2f}min)')

        bad_chips = find_high_std_chips(ret_dict, std_threshold = bad_std_threshold)
        average_std = calculate_average_std(ret_dict)
        save_to_json(dictionary = ret_dict,
                     filename = f'{save_path}/Chip_test_result.json')
        t_tot = time.time() - t_tot
        print(f'Time Total = {t_tot:.2f}s')
        np.random.seed(None)
        return bad_chips, average_std

    def chip_test_model(self, test_array = None, skip_array = None,
                        break_bad_chips = True, col_color = True,
                        plt_fig = False, save_fig = 1,
                        save_path = 'chip_test', bad_std_threshold = 0.07,
                        program_weight = True, prog_cycle = 80):

        t_tot = time.time()

        assert self.model is not None
        weight_dict = extract_weight_data(model = self.model)
        weight_arrays = weight_dict.keys()
        test_array_list = get_test_array_list(weight_arrays, test_array, skip_array)

        ret_dict = {}

        total_weight_size = calculate_total_weight_size(weight_dict, test_array_list)
        tested_weight_size = 0

        for array_idx, data in weight_dict.items():
            if array_idx in test_array_list:
                pass
            else:
                continue

            t_array = time.time()
            print(f'=========================')
            print(f'Testing Chip {array_idx} / {len(weight_dict) - 1}')
            ret_dict[f'array_{array_idx}'] = {}

            sdk = SDKArray(array_idx)

            for idx, data_dict in data.items():
                t_weight = time.time()
                weight_addr = data_dict['addr']
                weight_data = data_dict['data']

                if program_weight:
                    print(f'    ----------------------')
                    print(f'    Programing Weight {idx} / {len(data) - 1}')
                    sdk.set_weight_int4(weight_data,
                                        addr = weight_addr,
                                        prog_cycle = prog_cycle)

                input_rows = weight_addr[2]
                input_cols = 200
                input_matrix = np.random.randn(input_rows, input_cols)
                input_matrix[input_matrix < 0] = 0
                input_matrix_q, input_quant_scale = data_quant(input_matrix,
                                                               data_bit = 8,
                                                               isint = 1)

                adc_gain_best = simple_cim.ADC_auto_adjust(chip_idx = array_idx,
                                                           input_matrix = input_matrix_q,
                                                           addr = weight_addr,
                                                           weight = weight_data,
                                                           target_percent = [0.002, 0.001],
                                                           verbose = 0)

                print(f'    ----------------------')
                print(f'    Testing Weight {idx} / {len(data) - 1}')
                [mvm_result,
                 ADC_output,
                 ADC_scale] = simple_cim.mvm_calculate(array_idx,
                                                       weight = weight_data,
                                                       input_matrix = input_matrix_q,
                                                       use_simulator = 0,
                                                       addr = weight_addr,
                                                       it_time = adc_gain_best)

                mvm_result /= ADC_scale

                mvm_expected_int = np.dot(input_matrix_q.transpose(1, 0), weight_data)

                diff = mvm_expected_int - mvm_result
                data_range = mvm_expected_int.flatten().max() - mvm_expected_int.flatten().min()
                relative_std = diff.std() / data_range

                ret_dict[f'array_{array_idx}'][f'weight_{idx}'] = {}
                ret_dict[f'array_{array_idx}'][f'weight_{idx}']['diff_mean'] = diff.mean()
                ret_dict[f'array_{array_idx}'][f'weight_{idx}']['diff_std_relative'] = relative_std
                ret_dict[f'array_{array_idx}'][f'weight_{idx}']['weight_rows'] = weight_addr[2]
                ret_dict[f'array_{array_idx}'][f'weight_{idx}']['mvm_expected'] = mvm_expected_int
                ret_dict[f'array_{array_idx}'][f'weight_{idx}']['mvm_result'] = mvm_result

                t_weight = time.time() - t_weight
                print(f'array_{array_idx}||weight_{idx}||diff_std_relative = {relative_std}')
                print(f'Time for weight_{idx} = {t_weight:.2f}s')

                tested_weight_size += weight_data.size
                if relative_std >= bad_std_threshold and break_bad_chips:
                    remain_weight_data_size = sum([data_dict['data'].size for idx, data_dict in list(data.items())[idx + 1:]])
                    tested_weight_size += remain_weight_data_size
                    estimated_time = estimate_remaining_time(t_tot, tested_weight_size, total_weight_size)
                    print(f'Estimated remaining time: {estimated_time:.2f}s; ({estimated_time / 60:.2f}min)')
                    break

                estimated_time = estimate_remaining_time(t_tot, tested_weight_size, total_weight_size)
                print(f'Estimated remaining time: {estimated_time:.2f}s; ({estimated_time / 60:.2f}min)')

            t_array = time.time() - t_array
            print(f'Time for Array {array_idx} = {t_array:.2f}s')
            plot_weight_test_results(ret_dict, array_idx, plt_fig, save_fig = save_fig, save_path = save_path, col_color = col_color)
        bad_chips = find_high_std_chips(ret_dict, std_threshold = bad_std_threshold)
        average_std = calculate_average_std(ret_dict)
        save_to_json(dictionary = ret_dict,
                     filename = f'{save_path}/Chip_test_{self.name}.json')
        t_tot = time.time() - t_tot
        print(f'Time for Chip Test Total = {t_tot:.2f}s')
        return bad_chips, average_std
