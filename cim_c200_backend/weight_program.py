'''
@author: Wang Ze
'''
from c200_sdk.sdk_array_newsystem import SDKArray
from cim_c200_backend.utils import *


def program_weight_in_model_old(model, prog_cyc, mapping_repeat):
    for i in range(mapping_repeat):
        for name, module in model.named_modules():
            if type(module) in reg_dict.chip_conv_layers:
                if hasattr(module, 'weight_mapping_info'):
                    w_qn, w_scale = module.weight_quant_noise()
                    w_qn = module.get_2d_weight(w_qn)
                    weight_mapping_info = module.weight_mapping_info
                    for key, weight_info in weight_mapping_info.items():
                        start_row = weight_info['start_row']
                        start_col = weight_info['start_col']
                        row_num = weight_info['row_num']
                        col_num = weight_info['col_num']
                        array_idx = weight_info['array_idx']
                        weight_addr = weight_info['weight_addr']
                        w_qn_in = w_qn[start_row:start_row + row_num, start_col:start_col + col_num]
                        w_qn_in = torch.round(w_qn_in).to(torch.int8)

                        sdk = SDKArray(array_idx)
                        mapping_success_rate = sdk.set_weight_int4(to_numpy(w_qn_in),
                                                                   addr = weight_addr,
                                                                   prog_cycle = prog_cyc,
                                                                   return_log = 1)


def program_weight_in_model(model, prog_cyc,
                            prog_array = None, skip_array = None):
    weight_dict = extract_weight_data(model = model)
    weight_arrays = weight_dict.keys()
    test_array_list = get_test_array_list(weight_arrays,
                                          test_array = prog_array,
                                          skip_array = skip_array)

    bad_arrays = []

    total_weight_size = calculate_total_weight_size(weight_dict, test_array_list)
    tested_weight_size = 0

    t_start = time.time()
    for array_idx, data in weight_dict.items():
        if array_idx in test_array_list:
            pass
        else:
            continue
        bad_array_flag = 0
        t_array = time.time()
        print(f'=========================')
        print(f'Programing Chip {array_idx} / {len(weight_dict) - 1}')

        sdk = SDKArray(array_idx)

        for idx, data_dict in data.items():
            t_weight = time.time()
            weight_addr = data_dict['addr']
            weight_data = data_dict['data']
            tested_weight_size += weight_data.size

            print(f'    ----------------------')
            print(f'    Testing Array {array_idx}')
            mapping_success_rate = sdk.set_weight_int4(weight_data,
                                                       addr = weight_addr,
                                                       prog_cycle = 1,
                                                       return_log = 1)

            if mapping_success_rate < 50:
                bad_array_flag = 1
                bad_arrays.append(array_idx)
                break
            else:
                print(f'    ----------------------')
                print(f'    Programing Weight {idx} / {len(data) - 1}')
                sdk.set_weight_int4(weight_data,
                                    addr = weight_addr,
                                    prog_cycle = prog_cyc)

            estimated_time = estimate_remaining_time(t_start, tested_weight_size, total_weight_size)
            print(f'Estimated remaining time: {estimated_time:.2f}s; ({estimated_time / 60:.2f}min)')

        t_array = time.time() - t_array
        print(f'Time for Array {array_idx} = {t_array:.2f}s')
        if bad_array_flag:
            print(f'Failed to Program Array {array_idx}')

    t_tot = time.time() - t_start
    print(f'Time Total = {t_tot:.2f}s')
    return bad_arrays
