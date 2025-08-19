'''
@author: Wang Ze
'''

import cim_layers.layers_qn_lsq_adda_cim as adda_cim

try:
    from c200_sdk.sdk_array_newsystem import SDKArray
except:
    pass
from cim_layers.quant_noise_utils import *
from cim_toolchain_utils.utils import *
import torch.nn.functional as F

from cim_layers.layers_utils_lsq import *
from cim_layers.layers_utils_adda import *


class Conv2d_lsq_144k(adda_cim.Conv2d_lsq_adda_cim):
    def gen_output_tensor(self, x_2d):
        batch_num = x_2d.shape[0]
        output_rows = x_2d.shape[2]
        output_concated = torch.zeros([batch_num,
                                       output_rows,
                                       self.out_channels],
                                      device = self.weight.device)
        return output_concated

    def cal_x_weight_block(self, x_expanded, w_qn):

        w_qn = self.get_2d_weight(w_qn)
        x_q_2d = self.unfold(x_expanded)

        output_concat = self.gen_output_tensor(x_q_2d)

        for key, weight_info in self.weight_mapping_info.items():
            array_idx = weight_info['array_idx']
            self.weight_addr = weight_info['weight_addr']
            self.sdk = SDKArray(array_idx)
            start_row = weight_info['start_row']
            start_col = weight_info['start_col']
            row_num = weight_info['row_num']
            col_num = weight_info['col_num']

            x_split = x_q_2d[:, start_row:start_row + row_num, :]

            w_qn_in = w_qn[start_row:start_row + row_num, start_col:start_col + col_num]

            x = self.cal_x_bitwise(x_split, w_qn_in)

            output_concat[:, :, start_col:start_col + col_num] += x
        return output_concat

    def cal_x_bitwise(self, x_expanded, w_qn):
        batch = x_expanded.shape[0]
        x_rows = x_expanded.shape[1]
        x_cols = x_expanded.shape[2]
        x_expanded = x_expanded.permute(0, 2, 1)

        x_expanded_144k = x_expanded.reshape(-1, x_rows)

        out_array = self.sdk.calculate(to_numpy(x_expanded_144k),
                                     self.weight_addr,
                                     it_time = round(self.adc_gain.data.item()))
        out_array = to_tensor(out_array, device = w_qn.device)

        out_array = out_array.reshape(batch, x_cols, -1)
        return out_array

    def forward(self, x):
        if self.use_FP:
            x = self._conv_forward(x, self.weight, bias = self.bias)

        else:
            if self.shape_info is None:
                self.get_shape_info(x)
            self.refresh_adc_params()

            x_q, in_scale = input_quant(self, x, isint = True)

            w_qn, w_scale = weight_quant_noise(self, isint = True)

            output_concated = self.cal_x_weight_block(x_q, w_qn)
            x = self.fold_output(output_concated)

            x_tar = self._conv_forward(x_q, w_qn, bias = None)

            x = x / w_scale / in_scale / self.adc_scale
            x_tar = x_tar / w_scale / in_scale

            x = (x - x_tar).detach() + x_tar

            if self.bias is not None:
                x += self.bias.view(1, -1, 1, 1)
            x, out_scale = output_quant(self, x, isint = False)
        return x


class Linear_lsq_144k(adda_cim.Linear_lsq_adda_cim):
    def weight_quant_noise(self):
        w_qn = self.weight
        w_scale = 1.0
        if self.weight_quant:
            if self.step_size_weight == 1:
                init_step_size = self.init_step_size(self.weight, self.weight_bit)
                self.step_size_weight.data = init_step_size
                print(f'Initialized Step Size for Weight: {self.step_size_weight}')

            w_qn, w_scale = weight_quant_lsq(data_float = self.weight,
                                             data_bit = self.weight_bit,
                                             step_size = self.step_size_weight,
                                             isint = True)
        return w_qn, w_scale

    def gen_output_tensor(self, x_2d):
        batch_num = x_2d.shape[0]
        output_concated = torch.zeros([batch_num,
                                       self.out_features],
                                      device = self.weight.device)
        return output_concated

    def cal_x_weight_block(self, x_expanded, w_qn):

        output_concat = self.gen_output_tensor(x_expanded)

        for key, weight_info in self.weight_mapping_info.items():
            array_idx = weight_info['array_idx']
            self.weight_addr = weight_info['weight_addr']
            self.sdk = SDKArray(array_idx)
            start_row = weight_info['start_row']
            start_col = weight_info['start_col']
            row_num = weight_info['row_num']
            col_num = weight_info['col_num']

            x_split = x_expanded[:, start_row:start_row + row_num]

            w_qn_in = w_qn.permute(1, 0)
            w_qn_in = w_qn_in[start_row:start_row + row_num, start_col:start_col + col_num]

            x = self.cal_x_bitwise(x_split, w_qn_in)

            output_concat[:, start_col:start_col + col_num] += x
        return output_concat

    def cal_x_bitwise(self, x_expanded, w_qn):
        out_array = self.sdk.calculate(to_numpy(x_expanded),
                                     self.weight_addr,
                                     it_time = round(self.adc_gain.data.item()))
        out_array = to_tensor(out_array, device = w_qn.device)
        return out_array

    def forward(self, x):
        if self.use_FP:
            x = F.linear(x, self.weight, bias = self.bias)

        else:
            self.refresh_adc_params()

            x_q, in_scale = input_quant(self, x, isint = True)

            w_qn, w_scale = weight_quant_noise(self, isint = True)

            x = self.cal_x_weight_block(x_q, w_qn)

            x_tar = F.linear(x_q, w_qn, bias = None)

            x = x / w_scale / in_scale / self.adc_scale
            x_tar = x_tar / w_scale / in_scale

            x = (x - x_tar).detach() + x_tar

            if self.bias is not None:
                x += self.bias.view(1, -1)
            x, out_scale = output_quant(self, x, isint = False)
        return x
