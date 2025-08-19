'''
@author: Wang Ze
'''

import onnx
import torch
import torch.nn.functional as F
import torch.optim as optim
from onnx import shape_inference
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn as nn
import cim_layers.register_dict as reg_dict
import cim_qn_train.layers_enhance as en

from cim_layers.layers_all import *
from cim_toolchain_utils.utils import *
from . import hybrid_train_tools as hbt
import sys
import inspect


class CIMToolChain():
    def __init__(self, model, device = None, device_ids = None, name = 'model'):
        self.model = model
        self.model_name = name

        self.weight_bit = 8

        self.input_bit = 8
        self.output_bit = 8

        self.input_quant = True
        self.output_quant = True
        self.weight_quant = True

        self.clamp_std = 0

        self.noise_scale = 0
        self.gain_noise_scale = 0
        self.offset_noise_scale = 0

        self.adc_bit = 8
        self.dac_bit = 8
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device
        self.device_ids = device_ids
        self.onnx_model = None

        self.para_list = ['weight_bit', 'input_bit', 'output_bit',
                          'input_quant', 'output_quant', 'weight_quant',
                          'clamp_std',
                          'noise_scale', 'gain_noise_scale', 'offset_noise_scale',
                          'adc_bit', 'dac_bit']
        self.assign_module_name()

    def export_onnx(self, input_data, onnx_path = None):

        def find_non_leaf_tensors(module):
            bad_attrs = []

            def _check(obj, prefix = ""):
                if isinstance(obj, torch.Tensor):
                    if not obj.is_leaf:
                        bad_attrs.append(prefix)
                elif isinstance(obj, dict):
                    for k, v in obj.items():
                        _check(v, f"{prefix}.{k}" if prefix else k)
                elif isinstance(obj, (list, tuple)):
                    for idx, item in enumerate(obj):
                        _check(item, f"{prefix}[{idx}]")
                elif hasattr(obj, '__dict__'):
                    for attr_name in dir(obj):
                        if attr_name.startswith("__"):
                            continue
                        try:
                            val = getattr(obj, attr_name)
                            _check(val, f"{prefix}.{attr_name}" if prefix else attr_name)
                        except Exception:
                            pass

            _check(module)
            return bad_attrs

        def make_leaf_tensor(obj, attr_path):
            parts = attr_path.split('.')
            current = obj
            for i, p in enumerate(parts[:-1]):
                if isinstance(current, dict):
                    if p in current:
                        current = current[p]
                    else:
                        raise AttributeError(f"Key '{p}' not found in dict at {'.'.join(parts[:i])}")
                else:
                    if hasattr(current, p):
                        current = getattr(current, p)
                    elif hasattr(current, '_modules') and p in current._modules:
                        current = current._modules[p]
                    else:
                        raise AttributeError(f"Attribute '{p}' not found in {current}")

            final_attr = parts[-1]
            leaf_tensor = getattr(current, final_attr)
            leaf_tensor = leaf_tensor.detach().clone()
            leaf_tensor.requires_grad_(False)
            setattr(current, final_attr, leaf_tensor)

        bad_tensors = find_non_leaf_tensors(self.model)

        for path in bad_tensors:
            make_leaf_tensor(self.model, path)

        model_ = copy.deepcopy(self.model)
        if onnx_path is None:
            onnx_path = self.gen_model_name()

        dir_path = os.path.dirname(onnx_path)

        if not os.path.exists(dir_path) and len(dir_path) > 0:
            os.makedirs(dir_path)

        self.revert_to_nn_layer(model = model_, verbose = True)
        model_.eval()

        def to_device(data):
            if isinstance(data, list):
                return [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in data]
            elif isinstance(data, tuple):
                return tuple(item.to(self.device) if isinstance(item, torch.Tensor) else item for item in data)
            elif isinstance(data, torch.Tensor):
                return data.to(self.device)
            else:
                return data

        input_data = to_device(input_data)
        model_path = f"{onnx_path}.onnx"
        torch.onnx.export(model_,
                          input_data,
                          model_path,
                          export_params = True,
                          opset_version = 11,
                          do_constant_folding = True,
                          input_names = ['modelInput'],
                          output_names = ['modelOutput'],

                          )
        inferred_model = onnx.load_model(model_path)
        inferred_model = shape_inference.infer_shapes(inferred_model)
        onnx.save(inferred_model, model_path)

    def revert_to_nn_layer(self,
                           model = None,
                           exclude_layers = None,
                           assign_layers = None,
                           verbose = True):

        to_replace = []
        if model is None:
            model = self.model

        for name, module in model.named_modules():
            if exclude_layers is not None:

                if name in exclude_layers:
                    continue
            elif assign_layers is not None:

                if name not in assign_layers:
                    continue

            if type(module) in reg_dict.custom_conv_layers:

                new_module = nn.Conv2d(
                    in_channels = module.in_channels,
                    out_channels = module.out_channels,
                    kernel_size = module.kernel_size,
                    stride = module.stride,
                    groups = module.groups,
                    padding = module.padding,
                    bias = (module.bias is not None),
                )
                new_module.weight = module.weight
                if module.bias is not None:
                    new_module.bias = module.bias
                self.copy_meta_info(module, new_module)
                to_replace.append((name, new_module))

            elif type(module) in reg_dict.custom_linear_layers:

                new_module = nn.Linear(
                    in_features = module.in_features,
                    out_features = module.out_features,
                    bias = (module.bias is not None),
                )
                new_module.weight = module.weight
                if module.bias is not None:
                    new_module.bias = module.bias
                self.copy_meta_info(module, new_module)
                to_replace.append((name, new_module))

        if verbose:
            print(f'\n=============================================')
            if len(to_replace) == 0:
                print(f'No Layer Reverted to NN Layer')

            for name, new_module in to_replace:
                print(f'Reverted to NN Layer: {name}')
                self.find_and_replace_module(model, name, new_module)
            print(f'=============================================\n')

        self.set_device()

    def assign_module_name(self):
        for name, module in self.model.named_modules():
            if type(module) in reg_dict.op_layers:
                module.name = name

    def train_model(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def model_info(self):
        return self.get_model_info()

    def get_model_info(self, weight_info = False, show_nn_layers = True):
        model_info_dict = {}
        for name, module in self.model.named_modules():
            if name == "":
                continue
            if type(module) in reg_dict.op_layers:
                if show_nn_layers and not type(module) in reg_dict.op_layers:
                    continue
                model_info_dict[name] = {}
                model_info_dict[name]['type'] = type(module).__name__
                model_info_dict[name]['layer_flag'] = getattr(module, 'layer_flag', 'regular_layer')
                weight = getattr(module, 'weight', None)
                if weight is not None:
                    weight_shape = weight.shape
                    model_info_dict[name]['weight_shape'] = list(weight_shape)
                if weight_info:
                    model_info_dict[name]['weights'] = weight

        return model_info_dict

    def get_parameters_count(self):
        param_count = {}
        for name, module in self.model.named_modules():
            if type(module) in reg_dict.op_layers:
                params = sum(p.numel() for p in module.parameters())
                if type(module) in reg_dict.op_layers:
                    param_count[type(module).__name__] = param_count.get(type(module).__name__, 0) + params
                param_count['total_parameters'] = param_count.get('total_parameters', 0) + params
        return param_count

    @property
    def layer_names(self):
        name_list = []
        for name, module in self.model.named_modules():
            if type(module) in reg_dict.op_layers:
                name_list.append(name)
        return name_list

    @property
    def customized_layers(self):
        name_dict = {}
        for name, module in self.model.named_modules():
            if type(module) in reg_dict.custom_layers:
                type_name = type(module).__name__
                if type_name not in name_dict:
                    name_dict[type_name] = []
                name_dict[type_name].append(name)
        return name_dict

    def forward_with_hooks(self, input_data):
        """
        前向传播并记录每层的输入特征图
        """
        self.layer_input_shape = {}
        hooks = []
        model_temp = copy.deepcopy(self.model)

        def hook_factory(name):
            def hook(module, input, output):
                self.layer_input_shape[name] = input[0].shape

            return hook

        for name, module in model_temp.named_modules():
            hook = hook_factory(name)
            hooks.append(module.register_forward_hook(hook))

        output = model_temp(input_data)

        for hook in hooks:
            hook.remove()

        return output

    def update_self_parameter(self,
                              param_dict
                              ):
        for key, value in param_dict.items():
            if key in self.para_list and value is not None:
                setattr(self, key, value)

    def update_layer_parameter(self,
                               update_layer_type_list,
                               **kwargs):
        update_flag = 0
        self.update_self_parameter(kwargs)

        tar_classes = []

        for cls in reg_dict.op_layers:
            for layer_type in update_layer_type_list:

                if layer_type in cls.__module__.split('.'):
                    tar_classes.append(cls)

        for module in self.model.modules():
            if type(module) in tar_classes:
                module.update_para(**kwargs)
                update_flag += 1

        if update_flag == 0:
            print(f'No Layer Params Updated. Program Ended.')
            exit(1)

    def get_qn_parameter(self):
        para_dict = {}
        for param in self.para_list:
            if hasattr(self, param):
                para_dict[param] = getattr(self, param)
        return para_dict

    def find_and_replace_module(self, parent, name, new_module):
        """递归查找并替换指定名称的模块"""
        attrs = name.split('.')
        for i, attr in enumerate(attrs):
            if i == len(attrs) - 1:

                setattr(parent, attr, new_module)
            else:

                parent = getattr(parent, attr)

    def convert_to_layers(self,
                          convert_layer_type_list,
                          tar_layer_type,
                          exclude_layers = None,
                          assign_layers = None,
                          **kwargs):

        for key, value in kwargs.items():
            if key in self.para_list:
                setattr(self, key, value)

        tar_conv_class = None
        for cls in reg_dict.conv_layers:
            if tar_layer_type in cls.__module__.split('.'):
                tar_conv_class = cls
                break

        tar_linear_class = None
        for cls in reg_dict.linear_layers:
            if tar_layer_type in cls.__module__.split('.'):
                tar_linear_class = cls
                break

        if tar_conv_class is None and tar_linear_class is None:
            raise ValueError(f"Invalid layer_type '{tar_layer_type}'. Ensure it is correct and registered.")

        if exclude_layers is not None and assign_layers is not None:
            raise ValueError("Either 'exclude_layers' or 'assign_ACIM_layers' should be provided, but not both.")

        to_replace = []
        for name, module in self.model.named_modules():
            if exclude_layers and name in exclude_layers:
                continue
            if assign_layers and name not in assign_layers:
                continue
            if type(module) in convert_layer_type_list:
                if 'conv' in type(module).__name__.lower():
                    new_module = tar_conv_class(
                        in_channels = module.in_channels,
                        out_channels = module.out_channels,
                        kernel_size = module.kernel_size,
                        stride = module.stride,
                        padding = module.padding,
                        groups = module.groups,
                        bias = (module.bias is not None),
                        **kwargs
                    )

                elif 'linear' in type(module).__name__.lower():
                    new_module = tar_linear_class(
                        in_features = module.in_features,
                        out_features = module.out_features,
                        bias = (module.bias is not None),
                        **kwargs
                    )
                else:
                    raise NotImplementedError
                new_module.weight = module.weight
                if module.bias is not None:
                    new_module.bias = module.bias
                self.copy_meta_info(module, new_module)
                self.copy_lsq_data(module, new_module)
                to_replace.append((name, new_module))

        print(f"\n=============================================")
        if not to_replace:
            print(f"No layers converted to {tar_layer_type} layers.")
        for name, new_module in to_replace:
            print(f"Converted to {tar_layer_type} Layer: {name}")
            self.find_and_replace_module(self.model, name, new_module)
        print(f"=============================================\n")
        self.set_device()

    def convert_to_modules(self,
                           convert_layer_type_list,
                           tar_layer_type,
                           exclude_layers = None,
                           assign_layers = None,
                           **kwargs):

        for key, value in kwargs.items():
            if key in self.para_list:
                setattr(self, key, value)

        tar_conv_class = None
        for cls in reg_dict.conv_layers:
            if tar_layer_type in cls.__module__.split('.'):
                tar_conv_class = cls
                break

        tar_linear_class = None
        for cls in reg_dict.linear_layers:
            if tar_layer_type in cls.__module__.split('.'):
                tar_linear_class = cls
                break

        if tar_conv_class is None and tar_linear_class is None:
            raise ValueError(f"Invalid layer_type '{tar_layer_type}'. Ensure it is correct and registered.")

        if exclude_layers is not None and assign_layers is not None:
            raise ValueError("Either 'exclude_layers' or 'assign_ACIM_layers' should be provided, but not both.")

        to_replace = []
        for name, module in self.model.named_modules():
            if exclude_layers and name in exclude_layers:
                continue
            if assign_layers and name not in assign_layers:
                continue
            if type(module) in convert_layer_type_list:
                if 'conv' in type(module).__name__.lower():
                    new_module = tar_conv_class(module, **kwargs)

                elif 'linear' in type(module).__name__.lower():
                    new_module = tar_linear_class(module, **kwargs)
                else:
                    raise NotImplementedError
                to_replace.append((name, new_module))

        print(f"\n=============================================")
        if not to_replace:
            print(f"No layers converted to {tar_layer_type} layers.")
        for name, new_module in to_replace:
            print(f"Converted to {tar_layer_type} Layer: {name}")
            self.find_and_replace_module(self.model, name, new_module)
        print(f"=============================================\n")
        self.set_device()

    def copy_meta_info(self, module, new_module):
        new_module.name = module.name
        if hasattr(module, 'layer_flag'):
            new_module.layer_flag = getattr(module, 'layer_flag')

    def copy_lsq_data(self, module, new_module):
        if hasattr(module, 'step_size_weight') and hasattr(new_module, 'step_size_weight'):
            new_module.step_size_weight.data = module.step_size_weight.data
            new_module.step_size_input.data = module.step_size_input.data
            new_module.step_size_output.data = module.step_size_output.data

    def add_enhance_layers(self, conv_groups = 4):
        to_replace = []

        for name, module in self.model.named_modules():
            if type(module) in reg_dict.custom_conv_layers:
                new_module = en.EnhanceLayerConv2d(module, groups = conv_groups)
                to_replace.append((name, new_module))
            elif type(module) in reg_dict.custom_linear_layers:
                new_module = en.EnhanceLayerLinear(module)
                to_replace.append((name, new_module))

        for name, new_module in to_replace:
            self.find_and_replace_module(self.model, name, new_module)

        self.set_device()
        self.assign_module_name()

    def add_enhance_branch(self, conv_groups = 1, rank_factor = 0.25):
        to_replace = []

        for name, module in self.model.named_modules():
            if type(module) in reg_dict.custom_conv_layers:
                new_module = en.EnhanceBranchConv2d(
                    original_conv = module,
                    groups = conv_groups
                )
                to_replace.append((name, new_module))

            elif type(module) in reg_dict.custom_linear_layers:
                new_module = en.EnhanceBranchLinear_LoR(original_linear = module,
                                                        rank_factor = rank_factor,
                                                        )
                to_replace.append((name, new_module))

        for name, new_module in to_replace:
            self.find_and_replace_module(self.model, name, new_module)

        self.set_device()
        self.assign_module_name()

    def add_enhance_branch_LoR(self,
                               conv_groups = 1,
                               rank_factor = 1 / 4,
                               relu = False,
                               sigmoid = True):
        to_replace = []

        for name, module in self.model.named_modules():
            if type(module) in reg_dict.custom_conv_layers:
                new_module = en.EnhanceBranchConv2d_LoR(original_conv = module,
                                                        relu = relu,
                                                        sigmoid = sigmoid,
                                                        groups = conv_groups,
                                                        rank_factor = rank_factor)
                to_replace.append((name, new_module))

            elif type(module) in reg_dict.custom_linear_layers:
                new_module = en.EnhanceBranchLinear_LoR(original_linear = module,
                                                        relu = relu,
                                                        sigmoid = sigmoid,
                                                        rank_factor = rank_factor)
                to_replace.append((name, new_module))

        for name, new_module in to_replace:
            self.find_and_replace_module(self.model, name, new_module)

        self.set_device()
        self.assign_module_name()

    def zero_qn_layers(self):
        for name, module in self.model.named_modules():
            if type(module) in (l_qn_lsq.Conv2d_qn_lsq, l_qn_lsq.Linear_qn_lsq):
                for p_name, param in module.named_parameters():
                    param.detach()
                    param.data = param.data * 0

    def zero_branch_layers(self):
        for name, module in self.model.named_modules():
            if getattr(module, 'layer_flag', None) == 'enhance_branch':
                for p_name, param in module.named_parameters():
                    param.detach()
                    param.data = param.data * 0

    def set_blend_factors(self, value = 0.5):
        logit_value = torch.log(torch.tensor(value) / (1 - torch.tensor(value)))
        for name, module in self.model.named_modules():
            if getattr(module, 'blend_factor', None) is not None:
                module.blend_factor.data.fill_(logit_value.item())

    def set_requires_grad(self, name, p_name, param, requires_grad):
        param.requires_grad = requires_grad
        action = "Froze" if not requires_grad else "Unfroze"
        print(f"{action} parameter: {name}-{p_name}")

    def freeze_adc_gain(self, requires_grad = False):
        for name, module in self.model.named_modules():
            if type(module) in reg_dict.adda_layers:
                for p_name, param in module.named_parameters():
                    if p_name == 'adc_gain':
                        self.set_requires_grad(name, p_name, param, requires_grad)

    def freeze_step_size(self,
                         freeze_in_s = True,
                         freeze_out_s = True,
                         freeze_w_s = True,
                         requires_grad = False):
        p_name_list = []
        if freeze_in_s:
            p_name_list.append('step_size_input')
        if freeze_out_s:
            p_name_list.append('step_size_output')
        if freeze_w_s:
            p_name_list.append('step_size_weight')

        for name, module in self.model.named_modules():
            if type(module) in reg_dict.adda_layers:
                for p_name, param in module.named_parameters():
                    if p_name in p_name_list:
                        self.set_requires_grad(name, p_name, param, requires_grad)

    def freeze_backbone(self):
        for name, module in self.model.named_modules():
            if type(module) in reg_dict.custom_layers:
                for p_name, param in module.named_parameters():
                    self.set_requires_grad(name, p_name, param, False)

    def freeze_adda_layers(self, requires_grad = False):
        for name, module in self.model.named_modules():
            if type(module) in reg_dict.adda_layers:
                for p_name, param in module.named_parameters():
                    self.set_requires_grad(name, p_name, param, requires_grad)

    def freeze_qn_layers(self, requires_grad = False):
        for name, module in self.model.named_modules():
            if type(module) in (l_qn_lsq.Conv2d_qn_lsq, l_qn_lsq.Linear_qn_lsq):
                for p_name, param in module.named_parameters():
                    self.set_requires_grad(name, p_name, param, requires_grad)

    def freeze_blend_factors(self, requires_grad = False):
        for name, module in self.model.named_modules():
            if getattr(module, 'blend_factor', None) is not None:
                self.set_requires_grad(name, 'blend_factor', module.blend_factor, requires_grad)

    def freeze_bn_layers(self, requires_grad = False):
        for layer in self.model.modules():
            if isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm3d):
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = requires_grad

    def freeze_enhance_branch(self, requires_grad = False):
        for name, module in self.model.named_modules():
            if getattr(module, 'layer_flag', None) == 'enhance_branch':
                for p_name, param in module.named_parameters():
                    self.set_requires_grad(name, p_name, param, requires_grad)

    def freeze_enhance_layer(self, requires_grad = False):
        for name, module in self.model.named_modules():
            if getattr(module, 'layer_flag', None) == 'enhance_layer':
                for p_name, param in module.named_parameters():
                    self.set_requires_grad(name, p_name, param, requires_grad)

    @staticmethod
    def get_step(d_range, cycles):
        step = (d_range[1] - d_range[0]) / (cycles - 1) if cycles > 1 else 0
        return step

    @staticmethod
    def get_step_exp(d_range, cycles):
        if cycles < 2:
            return [0]

        scale = np.linspace(2, 1, cycles - 1)
        scale = np.exp(scale - 1)

        normalized_scale = scale / scale.sum()
        step_list = normalized_scale * (d_range[1] - d_range[0])

        return step_list.tolist()

    def compare_model_weights(self, model1, model2):
        model1_state_dict = model1.state_dict()
        model2_state_dict = model2.state_dict()

        same_weights = []
        different_weights = []

        for key in model1_state_dict.keys():
            if key in model2_state_dict:
                if torch.equal(model1_state_dict[key], model2_state_dict[key]):
                    same_weights.append(key)
                else:
                    different_weights.append(key)
            else:
                different_weights.append(key)

        for key in model2_state_dict.keys():
            if key not in model1_state_dict:
                different_weights.append(key)

        print(f'----------------')
        print("Same Weights:")
        print(f'----------------')
        for key in same_weights:
            print(key)
        print(f'\n')
        print(f'----------------')
        print("Different Weights:")
        print(f'----------------')
        for key in different_weights:
            print(key)

        return {
            "same_weights": same_weights,
            "different_weights": different_weights
        }

    def forward_with_hooks_layer_flag(self, model, input_data, layer_flag = ['enhance_layer']):
        output_dict = {}
        hooks = []

        def hook_factory(name):
            def hook(module, input, output):
                if getattr(module, 'layer_flag', None) in layer_flag:
                    output_dict[name] = output

            return hook

        for name, module in model.named_modules():
            hook = hook_factory(name)
            hooks.append(module.register_forward_hook(hook))

        output = model(input_data)

        for hook in hooks:
            hook.remove()

        return output, output_dict

    def get_adc_config(self):
        adc_config_dict = {}
        adc_adjust_mode = 'gain'
        for name, module in self.model.named_modules():
            if hasattr(module, 'adc_gain'):
                adc_config_dict[name] = {}
                adc_gain = torch.clamp(module.adc_gain.data,
                                       min = module.adc_gain_min,
                                       max = module.adc_gain_max)
                if hasattr(module, 'adc_adjust_mode'):
                    adc_adjust_mode = module.adc_adjust_mode
                if adc_adjust_mode == 'gain':
                    adc_config_dict[name]['gain_level'] = adc_gain.round().item()
                else:
                    adc_config_dict[name]['current_range'] = (1 / adc_gain).round().item()
        return adc_config_dict

    def get_adda_adc_gain_dict(self):
        adc_gain_dict = {}
        for name, module in self.model.named_modules():
            if type(module) in reg_dict.adda_layers:
                adc_gain_module_dict = copy.deepcopy(module.adc_gain_dict)
                for key, val in adc_gain_module_dict.items():
                    adc_gain_module_dict[key] = int(round(val.data.item()))
                adc_gain_dict[name] = adc_gain_module_dict
        return adc_gain_dict

    def progressive_train(self,
                          qn_cycle,
                          update_layer_type_list,
                          start_cycle,
                          **kwargs):

        steps_dict = {}
        current_para_dict = {}

        for param_name, param_value in kwargs.items():
            if param_name.endswith('_range'):
                param = param_name.replace('_range', '')

                steps_dict[param] = self.get_step(param_value, qn_cycle)
                current_para_dict[param] = param_value[0]

        train_model_signature = inspect.signature(self.train_model)
        train_model_params = set(train_model_signature.parameters.keys())

        train_model_kwargs = {k: v for k, v in kwargs.items() if k in train_model_params}

        for cyc in range(qn_cycle):

            if cyc < start_cycle:
                for param, step in steps_dict.items():
                    current_para_dict[param] += step
                continue

            rounded_params = {
                key: round(value) if key != 'noise_scale' else value
                for key, value in current_para_dict.items()
            }

            print(f'\n')
            print(f'==============================================')
            print(f'Progressive Training')
            print(f'Layer Type = {update_layer_type_list}')
            print(f'Parameters:')
            for key, value in rounded_params.items():
                if key != 'noise_scale':
                    print(f'{key} = {value}')
                else:
                    print(f'{key} = {value:.3g}')
            print(f'==============================================')
            print(f'\n')
            self.update_layer_parameter(
                update_layer_type_list = update_layer_type_list,
                **rounded_params,
            )

            for param, step in steps_dict.items():
                current_para_dict[param] += step

            self.train_model(**train_model_kwargs)

    def set_device(self, device = None, device_ids = None):
        if device is None:
            device = self.device
        if device_ids is None:
            device_ids = self.device_ids
        self.device = device
        self.device_ids = device_ids
        self.model.to(device)
        print(f'set model to device: {device}')
        if device_ids is not None:
            self.model = nn.DataParallel(self.model, device_ids = device_ids)
            print(f'model.device_ids =  {self.model.device_ids}')

    def load_model(self, PATH, strict = True):
        checkpoint = torch.load(PATH, map_location = self.device)
        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        model.load_state_dict(checkpoint, strict = strict)
        print("✅ Model loaded!")

    def save_model(self, PATH):
        dir_path = os.path.dirname(PATH)
        if not os.path.exists(dir_path) and len(dir_path) > 0:
            os.makedirs(dir_path)

        model_to_save = self.model
        if isinstance(model_to_save, torch.nn.parallel.DistributedDataParallel):
            model_to_save = model_to_save.module

        torch.save(model_to_save.state_dict(), PATH)
        if hasattr(self, 'rank'):
            if self.rank == 0:
                print(f'✅ Model saved at {PATH}')
        else:
            print(f'✅ Model saved at {PATH}')

    def remove_module_prefix(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith('module.') else key
            new_state_dict[new_key] = value
        return new_state_dict

    def add_excluded_layers(self, excluded_layers):
        self.excluded_layers = excluded_layers

    def gen_model_name(self, str_suffix = None):
        model_name = f'{self.model_name}_w={self.weight_bit}b_i={self.input_bit}b_o={self.output_bit}b_n={self.noise_scale:.3g}'
        if str_suffix is not None:
            model_name = f'{model_name}_{str_suffix}'
        return model_name

    def plot_loss(self, loss_list, title = 'Training Loss', save_path = None):
        if save_path is None:
            save_path = self.gen_model_name('_loss')
            save_path = f'{save_path}.png'
        dir_path = os.path.dirname(save_path)
        if not os.path.exists(dir_path) and len(dir_path) > 0:
            os.makedirs(dir_path)
        plt.figure()
        plt.plot(loss_list)
        plt.title(title)
        plt.savefig(save_path)
        plt.close()

    def plot_model_parameters(self, model):
        blend_factor = {}
        weights_list = []
        biases_list = []
        for name, module in model.named_modules():
            if type(module) in (nn.Conv2d, nn.Linear):
                weights = module.weight.data.cpu().numpy()
                weights_list.append(weights)
                if module.bias is not None:
                    biases = module.bias.data.cpu().numpy()
                    biases_list.append(biases)

        all_weights = np.concatenate([w.flatten() for w in weights_list])
        all_biases = np.concatenate([b.flatten() for b in biases_list]) if biases_list else np.array([0])
        global_min = min(all_weights.min(), all_biases.min())
        global_max = max(all_weights.max(), all_biases.max())

        fig, axs = plt.subplots(len(weights_list), 1, figsize = (12, 6 * len(weights_list)))
        if len(weights_list) == 1:
            axs = [axs]

        i = 0
        for j, (name, module) in enumerate(model.named_modules()):
            if type(module) in (nn.Conv2d, nn.Linear):
                weights = module.weight.data.cpu().numpy()
                biases = module.bias.data.cpu().numpy() if module.bias is not None else None

                print(f'==============')
                print(f'Name: {name}')
                print(f'Weight Mean: {weights.mean()}')
                print(f'Weight Std: {weights.std()}')
                if biases is not None:
                    print(f'Bias Mean: {biases.mean()}')
                    print(f'Bias Std: {biases.std()}')
                else:
                    print('Biases: None')
                print(f'==============')

                axs[i].hist(weights.flatten(), bins = 50, alpha = 0.7, label = 'Weights', range = (global_min, global_max))
                if biases is not None:
                    axs[i].hist(biases.flatten(), bins = 50, alpha = 0.7, label = 'Biases', range = (global_min, global_max))
                axs[i].set_title(f'Distribution of Weights and Biases for {name}')
                axs[i].set_xlabel('Value')
                axs[i].set_ylabel('Frequency')
                axs[i].legend()
                i += 1
            if getattr(module, 'blend_factor', None) is not None:
                blend_factor[name] = F.sigmoid(module.blend_factor).detach().cpu().numpy()

        plt.tight_layout()
        plt.show()

        for key, value in blend_factor.items():
            print(f'{key}: {value}')
