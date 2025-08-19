'''
@author: Wang Ze
'''
import time

import cim_layers.register_dict as reg_dict
from cim_toolchain_utils.utils import *
from cim_layers.layers_utils_lsq import *
from cim_layers.layers_utils_adda import *


def extract_weight_data(model):
    weight_dict = {}
    for name, module in model.named_modules():
        if type(module) in reg_dict.chip_on_chip_layers:
            if hasattr(module, 'weight_mapping_info'):
                w_qn, w_scale = weight_quant_noise(module, isint = True)
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
                    weight_data = to_numpy(w_qn_in.to(torch.int8))

                    if array_idx not in weight_dict:
                        weight_dict[array_idx] = {}

                    split_number = len(weight_dict[array_idx])
                    weight_dict[array_idx][split_number] = {
                        'addr': weight_addr,
                        'data': weight_data
                    }

    return weight_dict


def calculate_total_weight_size(weight_dict, array_list):
    total_size = 0
    for array_idx in array_list:
        data = weight_dict[array_idx]
        for idx, data_dict in data.items():
            weight_data = data_dict['data']
            total_size += weight_data.size
    return total_size


def estimate_remaining_time(start_time, tested_size, total_size):
    elapsed_time = time.time() - start_time
    avg_time_per_unit = elapsed_time / tested_size
    remaining_size = total_size - tested_size
    estimated_remaining_time = avg_time_per_unit * remaining_size
    return estimated_remaining_time


def get_test_array_list(weight_arrays, test_array, skip_array):
    if test_array and skip_array:
        raise ValueError("Only one of test_array or skip_array should be provided.")

    weight_set = set(weight_arrays)
    test_set = set(test_array) if test_array else set()
    skip_set = set(skip_array) if skip_array else set()

    if test_set:
        result_set = weight_set.intersection(test_set)
    else:
        result_set = weight_set

    if skip_set:
        result_set = result_set.difference(skip_set)

    return sorted(result_set)


def plot_weight_test_results(ret_dict, array_idx, plt_fig, save_fig, save_path, col_color = False):
    data = ret_dict[f'array_{array_idx}']
    num_plots = len(data)
    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))

    fig, axs = plt.subplots(rows, cols, figsize = (cols * 6, rows * 6))

    if num_plots == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    for idx, (weight_key, results) in enumerate(data.items()):
        mvm_result = results['mvm_result']
        mvm_expected = results['mvm_expected']
        relative_std = results['diff_std_relative']

        title = f'{weight_key}, std = {relative_std:.3g}'
        scatter_plt_color(axs[idx],
                          mvm_result, mvm_expected,
                          title = title, col_color = col_color)

        axs[idx].plot(mvm_expected.flatten(), mvm_result.flatten(), 'o', markersize = 2, alpha = 0.1)
        range_max = max(mvm_result.max(), mvm_expected.max())
        range_min = min(mvm_result.min(), mvm_expected.min())
        range_abs = max(abs(range_min), range_max)
        axs[idx].set_xlim(-range_abs, range_abs)
        axs[idx].set_ylim(-range_abs, range_abs)
        axs[idx].set_ylabel('array_output')
        axs[idx].set_xlabel('array_output_target')
        axs[idx].plot([-range_abs, range_abs], [-range_abs, range_abs], color = 'red')
        axs[idx].axhline(0, color = 'green', linestyle = '--')
        axs[idx].axvline(0, color = 'green', linestyle = '--')
        axs[idx].set_aspect('equal', adjustable = 'box')
        axs[idx].set_title(title)

    plt.tight_layout()

    if save_fig:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, f'array_{array_idx}_scatter_plots.png')
        plt.savefig(file_path)
        print(f'{file_path} saved!')
    if plt_fig:
        plt.show()


def scatter_plt_color(ax, array_output, array_output_target, title = 'Scatter Plot',
                      range_abs = None, col_color = False):
    def generate_colors(num_colors, saturation = 0.9, lightness = 0.6):
        colors = []
        for i in np.linspace(0, 1, num_colors, endpoint = False):
            hue = i
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            colors.append(rgb)
        return colors

    if range_abs is None:
        range_max = max(array_output.max(), array_output_target.max())
        range_min = min(array_output.min(), array_output_target.min())
        range_abs = max(abs(range_min), range_max)

    if col_color:
        num_cols = array_output.shape[1]
        colors = generate_colors(num_cols)
        for col in range(num_cols):
            ax.scatter(array_output_target[:, col], array_output[:, col],
                       color = colors[col], s = 2, alpha = 0.7, label = f'Column {col}')
    else:
        ax.scatter(array_output_target.flatten(), array_output.flatten(),
                   s = 2, alpha = 0.1)

    ax.set_xlim(-range_abs, range_abs)
    ax.set_ylim(-range_abs, range_abs)
    ax.set_ylabel('array_output')
    ax.set_xlabel('array_output_target')
    ax.plot([-range_abs, range_abs], [-range_abs, range_abs], color = 'red')
    ax.axhline(0, color = 'green', linestyle = '--')
    ax.axvline(0, color = 'green', linestyle = '--')
    ax.set_aspect('equal', adjustable = 'box')
    ax.set_title(title)
