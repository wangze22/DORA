from pathlib import Path
from time import sleep

import numpy
from sys import stderr
import csv
import time

import os
import datetime
import json

try:
    from c200_sdk import base_api
    import fcntl
except:
    pass
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"


class SDKError(RuntimeError):

    def __init__(self, api, what, *args):
        super().__init__(f'SDK 调用 {api}{args!r} {what}时失败')


def load_csv(fn, dtype = 'int32'):
    assert fn
    return numpy.loadtxt(fn, dtype = dtype, delimiter = ',', ndmin = 2)


def save_csv(fn, data):
    assert fn
    numpy.savetxt(fn, data, delimiter = ',', fmt = '%d')


def log_proc(mode, current, total):
    print('@', mode, '...', current, '/', total, end = '\r', file = stderr, flush = True)


class ReRAM144KProfile:
    input_size = 576
    input_bits = 1
    output_size = 128
    output_bits = 3
    weight_bits = 3
    dtype = int

    @classmethod
    def to_dict(cls):
        return dict(input_size = cls.input_size,
                    input_bits = cls.input_bits,
                    output_size = cls.output_size,
                    output_bits = cls.output_bits,
                    weight_bits = cls.weight_bits,
                    dtype = cls.dtype.__name__ if isinstance(cls.dtype, type) else str(cls.dtype))


class SDKArray:
    PROFILE = ReRAM144KProfile
    NAME = 'sdk144k'

    map_error = 0
    map_count = 10
    map_ratio = .9

    W_LOW = 6
    MAX_LOW = 100

    MIN_IPW = 0
    LONG_IPW = 3200
    MAX_IPW = 1000000000

    SELECT_DELAY = 0.5

    c = 1

    dtype = 'int32'
    shape = PROFILE.input_size, PROFILE.output_size
    bits = PROFILE.input_bits, PROFILE.weight_bits, PROFILE.output_bits

    _sdk = None
    _id = None
    _idir = Path('input')
    _odir = Path('output')

    _it_time = 5
    _wl_pre = 40
    _wl_after = 255
    _ctrl_delay = 60

    @classmethod
    def status(cls):
        return dict(sdk = cls._sdk is not None, board = cls._id)

    @classmethod
    def connect(cls):
        if cls._sdk is None:
            cls._sdk = base_api.BaseAPI()

    @classmethod
    def select(cls, id):
        if cls._sdk is None:
            cls.connect()
        if cls._id == id:
            return

        cls._sdk.devInit()
        print('初始化')
        cls._sdk.selectChip(id)
        print('选片')
        sleep(cls.SELECT_DELAY)
        cls._id = id

    @classmethod
    def is_emulator(cls):
        return False

    @classmethod
    def set_intergration_time(cls, it_time, wl_pre, wl_after, ctrl_delay):
        cls._sdk.cfgReadPulse(ti = it_time, ts = wl_pre, tb = wl_after, te = ctrl_delay)

    @classmethod
    def set_wl_pre_time(cls, time):
        cls._sdk.cfgReadPulse(ts = time)

    @classmethod
    def set_wl_after_time(cls, time):
        cls._sdk.cfgReadPulse(tb = time)

    @classmethod
    def set_ctrl_delay_time(cls, time):
        cls._sdk.cfgReadPulse(te = time)

    def __init__(self, id, adc_cal_en = 0):

        self.lock_file = "/tmp/elememreram.lock"
        self.lock_file_obj = None

        with self.acquire_lock():
            self.select(id)
        self.id = id
        self.version = self._sdk.version
        self.adc_cal_cfg = None
        self.adc_cal_en = adc_cal_en
        if self.adc_cal_en > 0:
            self.weight_int4 = self.get_weight_int4(addr = [0, 0, 576, 128], verbose = 0)
            self.load_adc_calibration_cfg(cfg_filename = CURRENT_DIR + "adc_cal_config.json", adc_cal_en = self.adc_cal_en)

    def acquire_lock(self):
        self.lock_file_obj = open(self.lock_file, "w")
        try:
            fcntl.flock(self.lock_file_obj, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self.lock_file_obj
        except IOError:
            print("\n-------Error: SDKArray资源已被其他进程占用，初始化失败--------\n")
            return None

    def to_sdk_addr(self, addr):
        H, W = self.shape
        if addr is None:
            addr = (0, H, 0, W)
        else:
            y, x, h, w = addr
            if not (0 <= y < y + h <= H and 0 <= x < x + w <= W):
                raise ValueError(f'addr {addr} 超出了范围 {([1, H], [1, W])!r}')
            addr = (y, h, x, w)
        return addr

    def addr_check(self, addr):
        y, x, h, w = addr
        if not (0 <= y < y + h <= 576 and 0 <= x < x + w <= 128):
            raise ValueError(f'addr {addr} 超出了范围 {([1, 576], [1, 128])!r}')

    def get_weight_int4(self, addr = None, type_is_1t1r = 0, acc_times = 1, verbose = 1):
        """
        获取区域权重

        Args:
            addr: 读取阵列地址范围
            type_is_1t1r: 读取数据类型是否为1t1r, 否则为2t2r
            verbose: debug info

        Returns:
            output: weight, 数据范围[-8,7]
        """
        weight = self.get_weight(addr, type_is_1t1r, acc_times, verbose) - 8
        return weight

    def get_weight(self, addr = None, type_is_1t1r = 0, acc_times = 1, verbose = 1):
        """
        获取区域权重

        Args:
            addr: 读取阵列地址范围
            type_is_1t1r: 读取数据类型是否为1t1r, 否则为2t2r
            verbose: debug info

        Returns:
            output: weight, 数据范围[0,15]
        """

        with self.acquire_lock():
            self.addr_check(addr)
            rowstart = addr[0]
            colstart = addr[1]
            rowcount = addr[2]
            colcount = addr[3]
            self.select(self.id)

            self.set_intergration_time(63, 40, 80, 1)
            if verbose == True:
                print('')
            start = time.time()
            self._sdk.switch_hd_version(1)
            weight = self._sdk.elemem_read_weight(rowstart, rowcount, colstart, colcount, type_is_1t1r, acc_times)
            self._sdk.switch_hd_version(0)
            elapsed = (time.time() - start)
            weight = weight.astype(numpy.int8)
            if verbose == True:
                print("Time used: %s" % datetime.timedelta(seconds = elapsed))
            if verbose == True:
                print(weight.shape)

            if type_is_1t1r:
                shape = (rowcount * 2, colcount)
            else:
                shape = (rowcount, colcount)
            if weight.shape != shape:
                raise ValueError(f'输出权重维度 {weight.shape} 应为 {shape}')

            weight_0 = weight.copy()
            if acc_times > 1:
                weight_f = weight_0.astype(numpy.float)
                weight_f /= acc_times
                return weight_f
            else:
                return weight_0

    def set_weight_int4(self, weight, addr = None, form = False, quant = True, prog_cycle = 40,
                        verbose = 1, detail = False,
                        return_log = 0):
        """
        Map/Form 权重

        Args:
            weight: 权重，数据范围[-8,7]
            addr: 写入阵列地址范围
            from: 是否开启软件二次from
            prog_cycle: 循环写入次数
            verbose: debug info
        """
        if weight.max() > 7 or weight.min() < -8:
            raise ValueError(f'权重数据范围错误，应为-8到7之间')
        weight = weight + 8
        return self.set_weight(weight, addr, form, quant, prog_cycle, verbose, detail, return_log)

    def set_weight(self, weight, addr = None, form = False, quant = True, prog_cycle = 40,
                   verbose = 1, detail = False,
                   return_log = 0):
        """
        Map/Form 权重

        Args:
            weight: 权重，数据范围[0,15]
            addr: 写入阵列地址范围
            from: 是否启用from
            prog_cycle: 循环写入次数
            verbose: debug info
        """
        with self.acquire_lock():

            weight = numpy.asarray(weight).astype(numpy.int8)

            self.addr_check(addr)
            rowstart = addr[0]
            colstart = addr[1]
            rowcount = addr[2]
            colcount = addr[3]

            self.check_weight(weight, shape = (rowcount, colcount))
            print(f"{weight.shape = }")
            self.select(self.id)

            self.set_intergration_time(63, 40, 80, 1)
            cell_total_num = rowcount * colcount
            print(f"cell_total_num:{cell_total_num}")
            time_start = time.time()
            self._sdk.switch_hd_version(1)
            combine_pass_num, set_pass_num_arr = self._sdk.elemem_block_write_weight(weight, rowstart, colstart, prog_cycle)
            self._sdk.switch_hd_version(0)
            time_elapsed = (time.time() - time_start)

            print("Total time used: %s" % str(datetime.timedelta(seconds = time_elapsed)).split('.')[0])
            mapping_success_rate = 100.0 * (combine_pass_num) / cell_total_num
            print(f'Mapping success rate: {combine_pass_num} / {cell_total_num}, {mapping_success_rate:.4f} %')

        if self.adc_cal_en > 0:
            self.weight_int4[rowstart:rowstart + rowcount, colstart:colstart + colcount] = self.get_weight_int4(addr, verbose = 0)
        if return_log:
            return mapping_success_rate

    def calculate(self, input, addr = None, runner = None, it_time = 5, data_type = -1, expand_mode = 1, ret_mode = 0):
        '''
        计算一组输入数据, 可使用 runner 改变计算模式
        Args:
            it_time: int
                integration time
            data_type: int
                输入数据类型
                -1:检测input取值范围，自动设置data_type
                0:int1.5    [-1,1]
                1:int2      [-2,1]
                2:int3      [-4,3]
                3:int4      [-8,7]
                4:int5      [-16,15]
                5:int6      [-32,31]
                6:int7      [-64,63]
                7:int8      [-128,127]
            expand_mode: int
                data expand mode, 0:全展开; 1:位展开
            ret_mode:
                是否返回硬件展开累加前的原始数据
        Returns:
            output: int32
        '''

        if runner is None:
            func = self._calculate
        else:
            func = runner(self._calculate)
        return func(input, addr = addr, it_time = it_time, data_type = data_type, expand_mode = expand_mode, ret_mode = ret_mode)

    def _calculate(self, input, addr, it_time = 5, data_type = -1, expand_mode = 1, ret_mode = 0, wl_pre = 10, wl_after = 10, ctrl_delay = 1):
        '计算一组输入数据'
        with self.acquire_lock():
            self.addr_check(addr)
            rowstart = addr[0]
            colstart = addr[1]
            rowcount = addr[2]
            colcount = addr[3]

            input = numpy.asarray(input).astype(numpy.int8)
            H, W = self.shape
            if not numpy.issubdtype(input.dtype, numpy.integer):
                raise TypeError(f'输入数据类型应为整数类型, 而不是 {input.dtype}')
            if len(input.shape) == 1:
                input = input.reshape(1, input.size)
            if len(input.shape) != 2 or input.shape[1] != (rowcount):
                raise ValueError(f'输入数据维度 {input.shape} 错误')

            num = input.shape[0]
            self.select(self.id)
            self.set_intergration_time(it_time, wl_pre, wl_after, ctrl_delay)
            self._sdk.switch_hd_version(1)
            if self.adc_cal_cfg != None and self.adc_cal_en == 1:
                cur_weight = self.weight_int4[rowstart:rowstart + rowcount, colstart:colstart + colcount]
                weight_col_abs_means = numpy.mean(numpy.abs(cur_weight), axis = 0)
                weight_col_mean = weight_col_abs_means * (256 / 3.37)
                intercept = numpy.array(self.adc_cal_cfg["intercept"])
                adc_intercept = intercept * 131072
                self._sdk.set_clac_adc_cfg(it_time, adc_intercept, weight_col_mean, colstart, colcount)

            if data_type == -1:
                if abs(input).max() > 63:

                    data_type = 7
                elif abs(input).max() > 31:

                    data_type = 6
                elif abs(input).max() > 15:

                    data_type = 5
                elif abs(input).max() > 7:

                    data_type = 4
                elif abs(input).max() > 3:

                    data_type = 3
                elif abs(input).max() > 1:

                    data_type = 2
                else:
                    data_type = 1
            elif data_type == 0:
                data_type = 1

            output0 = self._sdk.elemem_calc_array(input, rowstart, rowcount, colstart, colcount, data_type, expand_mode, ret_mode)
            self._sdk.switch_hd_version(0)
            output = output0
            if ret_mode == 0:
                shape = (num, colcount)
                if output.shape != shape:
                    raise ValueError(f'输出数据维度 {output.shape} 应为 {shape}')

            return output

    def config_adc_calibration(self, adc_cfg, adc_cal_en = 0):
        self.adc_cal_cfg = adc_cfg
        self.adc_cal_en = adc_cal_en
        if adc_cal_en == 0:
            self._sdk.hw_adc_cal_switch(0)

    def load_adc_calibration_cfg(self, cfg_filename = "adc_cal_config.json", adc_cal_en = 1):
        if os.path.exists(cfg_filename):
            with open(cfg_filename, encoding = "utf-8") as f:
                adc_cal_cfg_all = json.load(f)
            self.config_adc_calibration(adc_cal_cfg_all[f"{self.id}"], adc_cal_en)
        else:
            self.config_adc_calibration(None, adc_cal_en)
            print("The adc calibration file does not exist")

    def row_negative_one_ratio(self, arr):
        num_rows = arr.shape[0]
        ratios = []
        for i in range(num_rows):
            row = arr[i]
            num_negatives = numpy.sum(row == -1)
            row_length = len(row)
            ratio = num_negatives / row_length
            ratios.append(ratio)
        return numpy.array(ratios)

    def get_weight_legacy(self, addr = None, verbose = 1):
        '获取区域权重'
        with self.acquire_lock():
            addr = self.to_sdk_addr(addr)

            rowstart = addr[0]
            rowcount = addr[1]
            colstart = addr[2]
            colcount = addr[3]
            self.select(self.id)

            self.set_intergration_time(63, 40, 255, 60)

            cnt = 0
            start = time.time()
            weight_matrix_pos = []
            if verbose == True:
                print('')
            for rowIdx in range(rowstart, rowstart + rowcount, 1):
                row_time_start = time.time()
                weight_row_pos = []
                weight_row_neg = []
                for colIdx in range(colstart, colstart + colcount, 1):
                    p = self._sdk.calcOneCell(rowIdx, colIdx)

                    weight_row_pos.append(p)

                weight_matrix_pos.append(weight_row_pos)
                row_time_used = time.time() - row_time_start
                if verbose == True:
                    print("\r%03d  %03d/576, row time used: %s" % (
                        cnt, rowIdx, datetime.timedelta(seconds = row_time_used)), flush = True)

            elapsed = (time.time() - start)
            if verbose == True:
                print("Time used: %s,cnt:%d" % (datetime.timedelta(seconds = elapsed), cnt), end = '')

            weight = numpy.array(weight_matrix_pos)
            if verbose == True:
                print(weight.shape)

            shape = (rowcount, colcount)
            if weight.shape != shape:
                raise ValueError(f'输出权重维度 {weight.shape} 应为 {shape}')
            return weight.copy()

    def check_weight(self, weight, shape):

        if not numpy.issubdtype(weight.dtype, numpy.integer):
            raise TypeError(f'权重矩阵的数据类型应为整数类型, 而不是 {weight.dtype}')
        if weight.shape != shape:
            raise ValueError(f'权重维度 {weight.shape} 应为 {shape}')

        N = 15
        if weight.max() > N or weight.min() < 0:
            raise ValueError(f'权重超出有效范围 {[0, N]}')

    def set_weight_ISPP(self, weight, addr = None, form = False, quant = True, verbose = 1,
                        return_log = 0):
        'Map/Form 权重'
        assert quant
        weight = numpy.asarray(weight).astype(numpy.int8)

        addr = self.to_sdk_addr(addr)
        y1 = addr[0]
        y2 = addr[1]
        x1 = addr[2]
        x2 = addr[3]

        shape = (y2, x2)
        H, W = self.shape
        self.check_weight(weight, shape)
        print(weight.shape)
        self.select(self.id)

        self.set_intergration_time(63, 40, 255, 60)

        auto_recovery_times = 0
        time_start = time.time()
        cell_total_num = 0
        cell_pass_num = 0
        cell_unchanged_num = 0
        for colIdx in range(x1, x1 + x2, 1):
            for rowIdx in range(y1, y1 + y2, 1):

                input_target_value = weight[rowIdx - y1][colIdx - x1]
                if input_target_value == 0:
                    input_target_value = 8

                if input_target_value < 0 or input_target_value > 15:
                    pass
                else:
                    cell_total_num = cell_total_num + 1
                    while True:
                        try:

                            r = self.program(rowIdx, colIdx, input_target_value, verbose = verbose)
                            if r:
                                cell_pass_num = cell_pass_num + 1
                                if r == 2:
                                    cell_unchanged_num = cell_unchanged_num + 1
                            else:
                                pass
                            break
                        except Exception as ex:
                            print('ERROR:[%3d, %3d] Exception, trying to reconnect...' % (rowIdx, colIdx))

                            time.sleep(1)

                            s = 'retry:(%d,%d)\n' % (rowIdx, colIdx)
                            print(s)
                            print(ex)
                            self.write_string('crash.txt', s)
                            reconnect_flag = False
                            raise ValueError(f'{ex}')
                            break

        time_elapsed = (time.time() - time_start)
        mapping_success_rate = 100.0 * (cell_pass_num) / cell_total_num
        operation_success_rate = 100.0 * (cell_pass_num - cell_unchanged_num) / (cell_total_num -
                                                                                 cell_unchanged_num)
        print("Total time used: %s" % (datetime.timedelta(seconds = time_elapsed)))
        print('Mapping success rate: %d / %d, %.4f%%' % (
            cell_pass_num, cell_total_num, 100.0 * (cell_pass_num) / cell_total_num))
        print('Operation success rate: %d / %d, %.4f%%' % (cell_pass_num - cell_unchanged_num,
                                                           cell_total_num - cell_unchanged_num,
                                                           100.0 * (cell_pass_num - cell_unchanged_num) / (
                                                                   cell_total_num - cell_unchanged_num)))
        print('Auto recovery times: %d' % (auto_recovery_times), flush = True)
        if return_log:
            return mapping_success_rate, operation_success_rate

    def set_weight_legacy(self, weight, addr = None, form = False, quant = True, prog_cycle = 5,
                          verbose = 1, detail = False,
                          return_log = 0):
        'Map/Form 权重'
        with self.acquire_lock():
            assert quant
            weight = numpy.asarray(weight).astype(numpy.int8)
            weight[weight == 0] = 8
            addr = self.to_sdk_addr(addr)
            y1 = addr[0]
            y2 = addr[1]
            x1 = addr[2]
            x2 = addr[3]

            shape = (y2, x2)
            H, W = self.shape
            self.check_weight(weight, shape)
            print(weight.shape)
            self.select(self.id)

            self.set_intergration_time(63, 40, 255, 60)

            auto_recovery_times = 0
            time_start = time.time()
            cell_total_num = y2 * x2
            cell_pass_num = 0
            cell_unchanged_num = 0
            cell_processed = 0
            mapping_success_rate_dict = {}
            operation_success_rate_dict = {}
            mapping_success_rate_dict['ISPP'] = []
            mapping_success_rate_dict['CDPP'] = []
            operation_success_rate_dict['ISPP'] = []
            operation_success_rate_dict['CDPP'] = []
            for cycle in range(1, prog_cycle + 1):
                print(f'\n')
                print(f'=======================================')
                print(f'Cycle = {cycle}')
                print(f'=======================================')
                if cycle == prog_cycle:
                    method = 'ISPP'
                else:
                    method = 'CDPP'

                for colIdx in range(x1, x1 + x2, 1):
                    for rowIdx in range(y1, y1 + y2, 1):
                        cell_processed += 1
                        input_target_value = weight[rowIdx - y1][colIdx - x1]
                        while True:
                            try:

                                if method == 'ISPP':
                                    r = self.program(rowIdx, colIdx, input_target_value, verbose = verbose)
                                else:
                                    r = self.program_CDPP(rowIdx, colIdx, input_target_value,
                                                          verbose = verbose)
                                if r:
                                    cell_pass_num = cell_pass_num + 1
                                    if r == 2:
                                        cell_unchanged_num = cell_unchanged_num + 1
                                print(f'\rProgress at {cell_processed / cell_total_num * 100:.2f} %', end = '')
                                break
                            except Exception as ex:
                                print('ERROR:[%3d, %3d] Exception, trying to reconnect...' % (rowIdx, colIdx))

                                time.sleep(1)

                                s = 'retry:(%d,%d)\n' % (rowIdx, colIdx)
                                print(s)
                                print(ex)
                                self.write_string('crash.txt', s)
                                reconnect_flag = False
                                raise ValueError(f'{ex}')
                                break

                time_elapsed = (time.time() - time_start)
                mapping_success_rate = 100.0 * (cell_pass_num) / cell_total_num

                if cell_total_num - cell_unchanged_num > 0:
                    operation_success_rate = 100.0 * (cell_pass_num - cell_unchanged_num) / (cell_total_num -
                                                                                             cell_unchanged_num)
                else:
                    operation_success_rate = 100.0
                print("Total time used: %s" % str(datetime.timedelta(seconds = time_elapsed)).split('.')[0])
                print(f'Mapping success rate: {cell_pass_num} / {cell_total_num}, {mapping_success_rate:.4f}')
                print(f'Operation success rate: {cell_pass_num - cell_unchanged_num} / '
                      f'{cell_total_num - cell_unchanged_num}, {operation_success_rate:.4f}')

                mapping_success_rate_dict[f'{method}'].append(mapping_success_rate)
                operation_success_rate_dict[f'{method}'].append(operation_success_rate)

                cell_pass_num = 0
                cell_unchanged_num = 0
                cell_processed = 0

            if return_log:
                return mapping_success_rate_dict, operation_success_rate_dict

    def write_string(self, path, data):
        if isinstance(data, str):
            file = open(path, 'a')
        file.write(data)
        file.close()

    def program(self, rowIdx, colIdx, input_target_value, verbose = 0):

        if input_target_value < 0 or input_target_value > 15:
            print('invalid input value for mapping!')
            return 0
        else:
            rowIdx = int(rowIdx)
            colIdx = int(colIdx)
            target_adc = input_target_value
            current_res_calc = self._sdk.calcOneCell(rowIdx, colIdx)
            if verbose:
                print('[%s]' % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")), end = ' ')
                print('[%03d, %03d] |' % (rowIdx, colIdx), end = ' ')
                print('[CALC %02d' % (current_res_calc), end = ' ')
                print('<POS %02d' % (self._sdk.readOneCell(rowIdx, colIdx, 'POS')), end = ' ')
                print('NEG %02d>]' % (self._sdk.readOneCell(rowIdx, colIdx, 'NEG')), end = ' ', flush = True)
                print('-->', end = ' ')
                print('[TAR %02d]' % (target_adc), end = ' ', flush = True)

            if current_res_calc == target_adc:
                if verbose:
                    print('PASS', flush = True)
                return 2
            else:
                r = self._sdk.map_single_device_2T2R(rowIdx, colIdx, target_adc, tolerance = 0, with_form = 1,
                                                     verbose = 0)

                if verbose:
                    print('[CALC %02d' % (self._sdk.calcOneCell(rowIdx, colIdx)), end = ' ')
                    print('<POS %02d' % (self._sdk.readOneCell(rowIdx, colIdx, 'POS')), end = ' ')
                    print('NEG %02d>]' % (self._sdk.readOneCell(rowIdx, colIdx, 'NEG')), end = ' ',
                          flush = True)
                if r:
                    if verbose:
                        print(' SUCC!\n', end = '', flush = True)
                    return 1
                else:
                    if verbose:
                        print(' FAIL!\n', end = '', flush = True)
                    return 0

    def program_CDPP(self, rowIdx, colIdx, input_target_value, verbose = 0, detail = False):

        rowIdx = int(rowIdx)
        colIdx = int(colIdx)
        if input_target_value < 0 or input_target_value > 15:
            print('invalid input value for mapping!')
            return 0
        elif input_target_value < 7 or input_target_value > 9:
            target_adc = input_target_value
            current_res_calc = self._sdk.calcOneCell(rowIdx, colIdx)
            if verbose:
                print('[%s]' % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")), end = ' ')
                print('[%03d, %03d] |' % (rowIdx, colIdx), end = ' ')
                print('[CALC %02d' % (current_res_calc), end = ' ')
                print('<POS %02d' % (self._sdk.readOneCell(rowIdx, colIdx, 'POS')), end = ' ')
                print('NEG %02d>]' % (self._sdk.readOneCell(rowIdx, colIdx, 'NEG')), end = ' ', flush = True)
                print('-->', end = ' ')
                print('[TAR %02d]' % (target_adc), end = ' ', flush = True)

            if current_res_calc == target_adc:
                if verbose:
                    print('PASS', flush = True)
                return 2
            else:
                r = self._sdk.map_single_device_2T2R_POR(rowIdx, colIdx, target_adc, tolerance = 0,
                                                         with_form = 1, verbose = 0)
                if verbose:
                    print('[CALC %02d' % (self._sdk.calcOneCell(rowIdx, colIdx)), end = ' ')
                    print('<POS %02d' % (self._sdk.readOneCell(rowIdx, colIdx, 'POS')), end = ' ')
                    print('NEG %02d>]' % (self._sdk.readOneCell(rowIdx, colIdx, 'NEG')), end = ' ',
                          flush = True)
                if r:
                    if verbose:
                        print(' SUCC!\n', end = '', flush = True)
                    return 1
                else:
                    if verbose:
                        print(' FAIL!\n', end = '', flush = True)
                    return 0
        else:
            target_adc = input_target_value
            current_res_calc = self._sdk.calcOneCell(rowIdx, colIdx)
            if verbose:
                print('[%s]' % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")), end = ' ')
                print('[%03d, %03d] |' % (rowIdx, colIdx), end = ' ')
                print('[CALC %02d' % (current_res_calc), end = ' ')
                print('<POS %02d' % (self._sdk.readOneCell(rowIdx, colIdx, 'POS')), end = ' ')
                print('NEG %02d>]' % (self._sdk.readOneCell(rowIdx, colIdx, 'NEG')), end = ' ', flush = True)
                print('-->', end = ' ')
                print('[TAR %02d]' % (target_adc), end = ' ', flush = True)

            if current_res_calc == target_adc:
                if verbose:
                    print('PASS', flush = True)
                return 2
            else:

                r = self._sdk.map_single_device_2T2R(rowIdx, colIdx, target_adc, tolerance = 0, with_form = 1,
                                                     verbose = 0)

                if verbose:
                    print('[CALC %02d' % (self._sdk.calcOneCell(rowIdx, colIdx)), end = ' ')
                    print('<POS %02d' % (self._sdk.readOneCell(rowIdx, colIdx, 'POS')), end = ' ')
                    print('NEG %02d>]' % (self._sdk.readOneCell(rowIdx, colIdx, 'NEG')), end = ' ',
                          flush = True)
                if r:
                    if verbose:
                        print(' SUCC!\n', end = '', flush = True)
                    return 1
                else:
                    if verbose:
                        print(' FAIL!\n', end = '', flush = True)
                    return 0

    def calculate_legacy(self, input, addr = None, runner = None, it_time = 5):
        '计算一组输入数据, 可使用 runner 改变计算模式'
        with self.acquire_lock():
            if runner is None:
                func = self._calculate_legacy
            else:
                func = runner(self._calculate_legacy)
            return func(input, addr = addr, it_time = it_time)

    def _calculate_legacy(self, input, addr, it_time = 5, wl_pre = 10, wl_after = 10, ctrl_delay = 10):
        '计算一组输入数据'

        addr = self.to_sdk_addr(addr)
        y1, y2, x1, x2 = addr
        input = numpy.asarray(input).astype(numpy.int8)
        H, W = self.shape
        if not numpy.issubdtype(input.dtype, numpy.integer):
            raise TypeError(f'输入数据类型应为整数类型, 而不是 {input.dtype}')
        if len(input.shape) == 1:
            input = input.reshape(1, input.size)
        if len(input.shape) != 2 or input.shape[1] != (y2):
            raise ValueError(f'输入数据维度 {input.shape} 错误')

        num = input.shape[0]
        self.select(self.id)
        self.set_intergration_time(it_time, wl_pre, wl_after, ctrl_delay)

        output = self._sdk.calc_array(addr, input)

        shape = (num, x2)
        if output.shape != shape:
            raise ValueError(f'输出数据维度 {output.shape} 应为 {shape}')
        return output

    def reset_chips_to_zeros_legacy(self, board_index):
        with self.acquire_lock():
            self.select(self.id)
            total_cell_num = 0
            action_cell_num = 0
            fail_cell_num = 0

            self.set_intergration_time(63, 40, 255, 60)

            auto_recovery_times = 0

            time_start = time.time()
            for repeat in range(3):
                for rowIdx in range(0, 576):
                    for colIdx in range(0, 128):
                        while True:
                            try:
                                total_cell_num = total_cell_num + 1

                                current_res_calc = self._sdk.calcOneCell(rowIdx, colIdx)
                                current_res_pos = self._sdk.readOneCell(rowIdx, colIdx, 'POS')
                                current_res_neg = self._sdk.readOneCell(rowIdx, colIdx, 'NEG')
                                print('[%s]' % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")),
                                      end = ' ')
                                print('%03d' % (board_index), end = ' ')
                                print('%03d' % (repeat), end = ' ')
                                print('[%03d, %03d]' % (rowIdx, colIdx), end = ' ')
                                print('CALC %02d' % (current_res_calc), end = ' ')
                                print('<POS %02d' % (current_res_pos), end = ' ')
                                print('NEG %02d>' % (current_res_neg), end = ' ', flush = True)

                                if (current_res_calc == 7 or current_res_calc == 8) and\
                                        (current_res_pos == 7 or current_res_pos == 8) and\
                                        (current_res_neg == 7 or current_res_neg == 8):
                                    print('--> pass', flush = True)
                                else:
                                    r_pos = 1
                                    r_neg = 1
                                    r_calc_1 = 1
                                    r_calc_2 = 1
                                    action_cell_num = action_cell_num + 1
                                    if current_res_pos > 8:
                                        r_pos = self._sdk.map_single_device(rowIdx, colIdx, 'POS', 7,
                                                                            tolerance = 1, verbose = 0)
                                    if current_res_neg > 8:
                                        r_neg = self._sdk.map_single_device(rowIdx, colIdx, 'NEG', 7,
                                                                            tolerance = 1, verbose = 0)
                                    current_res_calc = self._sdk.calcOneCell(rowIdx, colIdx)
                                    if current_res_calc >= 9:
                                        r_calc_1 = self._sdk.map_single_device_2T2R(rowIdx, colIdx, 8,
                                                                                    tolerance = 0, verbose = 0)
                                    if current_res_calc <= 6:
                                        r_calc_2 = self._sdk.map_single_device_2T2R(rowIdx, colIdx, 7,
                                                                                    tolerance = 0, verbose = 0)
                                    print('-->', end = ' ')
                                    print('CALC %02d' % (self._sdk.calcOneCell(rowIdx, colIdx)), end = ' ')
                                    print('<POS %02d' % (self._sdk.readOneCell(rowIdx, colIdx, 'POS')), end = ' ')
                                    print('NEG %02d>' % (self._sdk.readOneCell(rowIdx, colIdx, 'NEG')), end = '',
                                          flush = True)
                                    if r_pos and r_neg and r_calc_1 and r_calc_2:
                                        print('\n', end = '', flush = True)
                                    else:
                                        fail_cell_num = fail_cell_num + 1
                                        print(' FAIL!\n', end = '', flush = True)

                                break

                            except Exception as ex:

                                print('ERROR: [%3d] [%3d, %3d] Exception, trying to reconnect...' % (
                                    repeat, rowIdx, colIdx))

                                time.sleep(1)

                                s = 'retry:%d,(%d,%d)\n' % (repeat, rowIdx, colIdx)
                                print(s)
                                print(ex)
                                self.write_string('crash.txt', s)

                                reconnect_flag = False
                                raise ValueError(f'{ex}')

            time_elapsed = (time.time() - time_start)
            print("Time used: %s" % (datetime.timedelta(seconds = time_elapsed)))
            print('Total: %d, operation: %d, fail: %d, success rate: %.2f%%' % (
                total_cell_num, action_cell_num, fail_cell_num, (1.0 - fail_cell_num / action_cell_num) * 100.0))
