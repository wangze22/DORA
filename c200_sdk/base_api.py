import ctypes
import os
import time

import numpy as np

try:
    from pynq import MMIO
    from pynq.lib.iic import AxiIIC

except:
    pass
from pathlib import Path
import json

SDK_VERSION = "v2.8"

BASE_ADDR = 0xA0000000
ADDRESS_RANGE = 0x4000
BRAM_BASE_ADDR = 0xB0010000
BRAM_RANGE = 0x1000

CUR_CLOCK = 30
CLOCK_RATIO = CUR_CLOCK / 10

PULSE_UNIT_TIME = 100 / CLOCK_RATIO

REG0_ADDR = 0
REG1_ADDR = 1 * 4
REG2_ADDR = 2 * 4
REG3_ADDR = 3 * 4
REG4_ADDR = 4 * 4
REG5_ADDR = 5 * 4
REG6_ADDR = 6 * 4
REG7_ADDR = 7 * 4
REG8_ADDR = 8 * 4
REG9_ADDR = 9 * 4
REG10_ADDR = 10 * 4
REG11_ADDR = 11 * 4
REG12_ADDR = 12 * 4
REG13_ADDR = 13 * 4
REG14_ADDR = 14 * 4
REG15_ADDR = 15 * 4
REG51_ADDR = 51 * 4
REG52_ADDR = 52 * 4
REG55_ADDR = 55 * 4
REG56_ADDR = 56 * 4
REG57_ADDR = 57 * 4
REG58_ADDR = 58 * 4
REG59_ADDR = 59 * 4
REG65_ADDR = 65 * 4
REG66_ADDR = 66 * 4
REG67_ADDR = 67 * 4
REG75_ADDR = 75 * 4
REG76_ADDR = 76 * 4
REG77_ADDR = 77 * 4
REG80_ADDR = 80 * 4

VERSION_TOGGLE = 0x0274
HW_MODE_RESET = 0x0278

IRQ_PULSE_WIDTH = 0x027c
ROW_REARRANGE_EN = 0x0280

CAL_IT_TIME = 0x3070
ADC0_B_CALIB = 0x3080
ADC1_B_CALIB = 0x3084
ADC2_B_CALIB = 0x3088
ADC3_B_CALIB = 0x308c
ADC4_B_CALIB = 0x3090
ADC5_B_CALIB = 0x3094
ADC6_B_CALIB = 0x3098
ADC7_B_CALIB = 0x309c
HW_CAL_ADC_EN = 0x30a0

TOTAL_ROW = 1152
TOTAL_CHANNEL = 8
ONE_CHANNEL_ROW = 144
TOTAL_COL = 128

OP_CFG_DAC = 1
OP_CALC = 2
OP_FORM = 3
OP_SET = 4
OP_RESET = 5
OP_READ = 6
OP_SELECT_CHIP = 9
OP_CFG_CSN = 0x0A
OP_POWER_ON = 0x0F
OP_POWER_OFF = 0x1F

POS_DIR = 1
NEG_DIR = 2

current_dir = os.path.dirname(os.path.realpath(__file__)) + "/"
CLIB_PATH = current_dir + "libBaseApi.so"


def mySleep(delayTime):
    startTime = time.perf_counter()
    while time.perf_counter() - startTime < delayTime:
        pass


def DACVToReg(voltage):
    return int(voltage * 0xFFFF / 5)


def regToCurrent(regVal):
    current = 3125 * regVal // 0xFFFF

    return current


def currentToReg(valmA):
    val = valmA * 0xFFFF // 3125
    return val


class DIN():
    def __init__(self, regAddr = 0, selectedBitMap = 0, actualBitMap = 0):
        self.cfgPara(regAddr, selectedBitMap, actualBitMap)

    def cfgPara(self, regAddr, selectedBitMap, actualBitMap = 0):
        self.regAddr = regAddr
        self.selectedBitMap = selectedBitMap
        self.actualBitMap = actualBitMap


class BaseAPI():
    version = SDK_VERSION

    def __init__(self):
        self.mmio = MMIO(BASE_ADDR, ADDRESS_RANGE)
        self.bram_mmio = MMIO(BRAM_BASE_ADDR, BRAM_RANGE)
        self.DIN0 = [DIN() for _ in range(ONE_CHANNEL_ROW)]
        self.DIN1 = [DIN() for _ in range(ONE_CHANNEL_ROW)]
        self.DIN2 = [DIN() for _ in range(ONE_CHANNEL_ROW)]
        self.DIN3 = [DIN() for _ in range(ONE_CHANNEL_ROW)]
        self.DIN4 = [DIN() for _ in range(ONE_CHANNEL_ROW)]
        self.DIN5 = [DIN() for _ in range(ONE_CHANNEL_ROW)]
        self.DIN6 = [DIN() for _ in range(ONE_CHANNEL_ROW)]
        self.DIN7 = [DIN() for _ in range(ONE_CHANNEL_ROW)]
        self.cfgDIN()
        self.DINArr = (self.DIN0, self.DIN1, self.DIN2, self.DIN3, self.DIN4,
                       self.DIN5, self.DIN6, self.DIN7)
        self.dictIic3 = {'phys_addr': 0xA0004000, 'addr_range': 0x1000}
        self.dictIic4 = {'phys_addr': 0xA0005000, 'addr_range': 0x1000}
        self.dictIic5 = {'phys_addr': 0xA0006000, 'addr_range': 0x1000}
        self.clib = ctypes.cdll.LoadLibrary(CLIB_PATH)
        self.clib.ElememDev_Init()

        self.irq_flag = 0
        self.helium_cfg = None

    def __del__(self):

        self.clib.ElememDev_DeInit()

    def conv_pulse(self, pulse):
        pulse = int((pulse + 1) * CLOCK_RATIO) - 1
        if pulse > 255:
            pulse = 255
        return pulse

    def switch_hd_version(self, on):
        self.writeReg(VERSION_TOGGLE, on)

    def adc_debug_mode(self, on):
        if on:
            self.writeReg(HW_MODE_RESET, 0)
        else:
            self.writeReg(HW_MODE_RESET, 2)

    def clearAllActualBitMap(self):
        for dinNum in self.DINArr:
            for din in dinNum:
                din.actualBitMap = 0

    def cfgDIN(self):
        reg = REG51_ADDR
        for i in range(TOTAL_ROW // 32):
            self.DIN0[0 + 4 * i].cfgPara(reg, 0x01000000)
            self.DIN0[1 + 4 * i].cfgPara(reg, 0x00010000)
            self.DIN0[2 + 4 * i].cfgPara(reg, 0x00000100)
            self.DIN0[3 + 4 * i].cfgPara(reg, 0x00000001)

            self.DIN1[0 + 4 * i].cfgPara(reg, 0x02000000)
            self.DIN1[1 + 4 * i].cfgPara(reg, 0x00020000)
            self.DIN1[2 + 4 * i].cfgPara(reg, 0x00000200)
            self.DIN1[3 + 4 * i].cfgPara(reg, 0x00000002)

            self.DIN2[0 + 4 * i].cfgPara(reg, 0x04000000)
            self.DIN2[1 + 4 * i].cfgPara(reg, 0x00040000)
            self.DIN2[2 + 4 * i].cfgPara(reg, 0x00000400)
            self.DIN2[3 + 4 * i].cfgPara(reg, 0x00000004)

            self.DIN3[0 + 4 * i].cfgPara(reg, 0x08000000)
            self.DIN3[1 + 4 * i].cfgPara(reg, 0x00080000)
            self.DIN3[2 + 4 * i].cfgPara(reg, 0x00000800)
            self.DIN3[3 + 4 * i].cfgPara(reg, 0x00000008)

            self.DIN4[0 + 4 * i].cfgPara(reg, 0x10000000)
            self.DIN4[1 + 4 * i].cfgPara(reg, 0x00100000)
            self.DIN4[2 + 4 * i].cfgPara(reg, 0x00001000)
            self.DIN4[3 + 4 * i].cfgPara(reg, 0x00000010)

            self.DIN5[0 + 4 * i].cfgPara(reg, 0x20000000)
            self.DIN5[1 + 4 * i].cfgPara(reg, 0x00200000)
            self.DIN5[2 + 4 * i].cfgPara(reg, 0x00002000)
            self.DIN5[3 + 4 * i].cfgPara(reg, 0x00000020)

            self.DIN6[0 + 4 * i].cfgPara(reg, 0x40000000)
            self.DIN6[1 + 4 * i].cfgPara(reg, 0x00400000)
            self.DIN6[2 + 4 * i].cfgPara(reg, 0x00004000)
            self.DIN6[3 + 4 * i].cfgPara(reg, 0x00000040)

            self.DIN7[0 + 4 * i].cfgPara(reg, 0x80000000)
            self.DIN7[1 + 4 * i].cfgPara(reg, 0x00800000)
            self.DIN7[2 + 4 * i].cfgPara(reg, 0x00008000)
            self.DIN7[3 + 4 * i].cfgPara(reg, 0x00000080)
            reg = reg - 4

    def writeReg(self, addr, value):
        self.mmio.write(addr, int(value))

    def readReg(self, addr):
        return self.mmio.read(addr)

    def writeBram(self, addr, value):
        self.bram_mmio.write(addr, int(value))

    def readBram(self, addr):
        return self.bram_mmio.read(addr)

    def getAdcValue(self, regVal, colIdx):
        idx = colIdx % 8
        adcVal = (regVal >> (idx * 4)) & 0xF
        return adcVal

    def selectOneCol(self, colIdx):
        assert 0 <= colIdx < TOTAL_COL
        self.writeReg(REG52_ADDR + 4 * 0, 0)
        self.writeReg(REG52_ADDR + 4 * 1, 0)
        self.writeReg(REG52_ADDR + 4 * 2, 0)
        self.writeReg(REG52_ADDR + 4 * 3, 0)
        regNum = colIdx // 32
        remainder = colIdx % 32
        regVal = 1 << (31 - remainder)
        self.writeReg(REG55_ADDR - 4 * regNum, regVal)

    def selectOneRow(self, rowIdx, POS_or_NEG):
        assert POS_or_NEG == 'POS' or POS_or_NEG == 'NEG'
        assert 0 <= rowIdx < (TOTAL_ROW // 2)
        regRow = 0
        channel = rowIdx * 2 // ONE_CHANNEL_ROW
        if POS_or_NEG == 'POS':
            regRow = rowIdx * 2 % ONE_CHANNEL_ROW + 1
            self.writeReg(REG14_ADDR, channel)
        else:
            regRow = rowIdx * 2 % ONE_CHANNEL_ROW + 2
            self.writeReg(REG14_ADDR, channel | 0x80000000)
        self.writeReg(REG13_ADDR, regRow)

    def waitFlagBit(self, addr, bitPosition, delay, bitSet = True):

        for i in range(delay):
            ret = self.readReg(addr)
            if bitSet:
                if ret & (1 << bitPosition):
                    return True
            else:
                if not (ret & (1 << bitPosition)):
                    return True
        return False

    def waitOpFinish(self, op):
        if (op == 0xf) or (op == 0x1f):
            bitPosition = 8
        else:
            bitPosition = op
        ret = self.waitFlagBit(REG0_ADDR, bitPosition, 100000, True)
        if not ret:
            raise TimeoutError('The completion flag is not set')

    def clearOp(self, op):
        v = 0
        if (op == 0xf) or (op == 0x1f):
            v = 0xFF
            bitPosition = 8
        else:
            v = op | 0x10
            bitPosition = op
        self.writeReg(REG3_ADDR, v)
        ret = self.waitFlagBit(REG0_ADDR, bitPosition, 100000, False)
        if not ret:
            raise TimeoutError('The completion flag is not reset')

    def opFlow(self, op):

        self.writeReg(REG3_ADDR, op)

        self.writeReg(REG1_ADDR, 1)

        self.waitOpFinish(op)

        self.clearOp(op)

        self.writeReg(REG1_ADDR, 0)

    def oneCellOp(self, op, rowIdx, colIdx, POS_or_NEG):

        assert POS_or_NEG == 'POS' or POS_or_NEG == 'NEG'

        self.selectOneRow(rowIdx, POS_or_NEG)

        self.selectOneCol(colIdx)

        if op == OP_CALC:
            self.writeReg(REG11_ADDR, 8 | POS_DIR | NEG_DIR)
        else:
            if POS_or_NEG == 'POS':
                self.writeReg(REG11_ADDR, 8 | POS_DIR)
            else:
                self.writeReg(REG11_ADDR, 8 | NEG_DIR)

        if op == OP_RESET:
            self.writeReg(REG15_ADDR, 0x80000001)
        else:
            self.writeReg(REG15_ADDR, 1)

        self.opFlow(op)

    def cfgCSN(self, channel, value, isReg):
        assert 0 <= channel <= 15
        if not isReg:
            value = currentToReg(value)
        self.writeReg(REG66_ADDR, value)
        self.writeReg(REG65_ADDR, channel)
        self.opFlow(OP_CFG_CSN)

    def cfgDAC(self, channel, value, isReg):
        assert 0 <= channel <= 15
        if not isReg:
            value = DACVToReg(value)
        self.writeReg(REG5_ADDR, value)
        regVal = (1 << 4) | channel
        self.writeReg(REG4_ADDR, regVal)
        self.opFlow(OP_CFG_DAC)

    def cfgAD5254(self, channel, regVal):
        pass

    def cfgVBLMid(self, value, isReg = False):
        self.cfgDAC(5, value, isReg)

    def cfgVBLNeg(self, value, isReg = False):
        self.cfgDAC(15, value, isReg)

    def cfgVBLPos(self, value, isReg = False):
        self.cfgDAC(14, value, isReg)

    def cfgVrefCompDn(self, value, isReg = False):
        self.cfgDAC(7, value, isReg)

    def cfgVrefCompUp(self, value, isReg = False):
        self.cfgDAC(4, value, isReg)

    def cfgVSLClamp(self, value, isReg = False):
        self.cfgDAC(0, value, isReg)

    def cfgVDDLV(self, value, isReg = False):
        self.cfgDAC(12, value, isReg)

    def cfgFormVBL(self, value, isReg = False):
        self.cfgDAC(6, value, isReg)

    def cfgFormVWL(self, value, isReg = False):
        self.cfgDAC(1, value, isReg)

    def cfgSetVBL(self, value, isReg = False):
        self.cfgDAC(13, value, isReg)

    def cfgSetVWL(self, value, isReg = False):
        self.cfgDAC(3, value, isReg)

    def cfgResetVSL(self, value, isReg = False):
        self.cfgDAC(2, value, isReg)

    def cfgResetVWL(self, value, isReg = False):
        self.cfgDAC(11, value, isReg)

    def cfgFloatBl(self):
        val = 1 | (1 << 4)
        self.writeReg(REG12_ADDR, val)

    def cfgClockMhz(self, clockMhz):
        self.writeReg(REG10_ADDR, 100 // clockMhz)

    def cfgFRSPulse(self, ts = 0x50, tp = 0x3C, te = 0x3C):
        '''
            for form/reset/set
            ____|----------------|______  BLPulse
            ____|-ts-|--tp--|-te-|______  WLPulse
        '''
        ts = self.conv_pulse(ts)
        tp = self.conv_pulse(tp)
        te = self.conv_pulse(te)
        regVal = tp | (ts << 8)
        self.writeReg(REG7_ADDR, regVal)
        self.writeReg(REG56_ADDR, te)

    def cfgReadPulse(self, ti = 0x20, ts = 0x28, tb = 0xFF, te = 0x3C):
        '''
            ____|--------------------|______  BLPulse
            ____|-ts-|----------|-te-|______  WLPulse
            ____|----|----tb--|_____________  CTRL_INTEG
            ____|-------------|-ti-|________  KEEP_INTEG
        '''
        ti = self.conv_pulse(ti)
        ts = self.conv_pulse(ts)
        tb = self.conv_pulse(tb)
        te = self.conv_pulse(te)
        regVal = tb | (ts << 8) | (ti << 16)
        self.writeReg(REG6_ADDR, regVal)
        self.writeReg(REG56_ADDR, te)

    def resetSystem(self):
        self.writeReg(REG2_ADDR, 1)
        self.writeReg(HW_MODE_RESET, 1)
        time.sleep(0.01)
        self.writeReg(REG2_ADDR, 0)
        self.writeReg(HW_MODE_RESET, 2)

    def powerOn(self):
        self.opFlow(OP_POWER_ON)

    def powerOff(self):
        self.opFlow(OP_POWER_OFF)

    def selectRows(self, rowData):
        assert len(rowData) == (TOTAL_ROW // 2)
        self.clearAllActualBitMap()
        for idx, data in enumerate(rowData):
            if data == 1:
                rowIndex = idx * 2
                channel = rowIndex // 144
                din = self.DINArr[channel]
                din[rowIndex % 144].actualBitMap = din[rowIndex %
                                                       144].selectedBitMap

        for i in range(TOTAL_ROW // 32):
            v = 0
            for j in range(TOTAL_CHANNEL):
                v = v | self.DINArr[j][0 + 4 * i].actualBitMap
                v = v | self.DINArr[j][1 + 4 * i].actualBitMap
                v = v | self.DINArr[j][2 + 4 * i].actualBitMap
                v = v | self.DINArr[j][3 + 4 * i].actualBitMap
                self.writeReg(self.DINArr[0][0 + 4 * i].regAddr, v)

    def selectInput(self, inputData):
        assert len(inputData) == (TOTAL_ROW // 2)
        self.clearAllActualBitMap()
        for idx, data in enumerate(inputData):
            if data == -1:
                rowIndex = idx * 2
                channel = rowIndex // 144
                din = self.DINArr[channel]
                din[rowIndex % 144].actualBitMap = din[rowIndex %
                                                       144].selectedBitMap
            elif data == 1:
                rowIndex = idx * 2 + 1
                channel = rowIndex // 144
                din = self.DINArr[channel]
                din[rowIndex % 144].actualBitMap = din[rowIndex %
                                                       144].selectedBitMap
            elif data == 0:
                pass
            else:
                raise ValueError('The element of inputData must be -1, 0, 1')

        for i in range(TOTAL_ROW // 32):
            v = 0
            for j in range(TOTAL_CHANNEL):
                v = v | self.DINArr[j][0 + 4 * i].actualBitMap
                v = v | self.DINArr[j][1 + 4 * i].actualBitMap
                v = v | self.DINArr[j][2 + 4 * i].actualBitMap
                v = v | self.DINArr[j][3 + 4 * i].actualBitMap
                self.writeReg(self.DINArr[0][0 + 4 * i].regAddr, v)

    def map_single_device(self,
                          rowIdx,
                          colIdx,
                          poldirstr,
                          target_adc,
                          tolerance = 0,
                          try_limit = 500,
                          strategy = 0,
                          verbose = 1):
        succ_flag = 0

        set_vol_lim = 3
        set_gate_lim = 1.8
        reset_vol_lim = 3.5
        reset_gate_lim = 4.5

        if not strategy:
            set_vol_start = 0.8
            set_vol_step = 0.05
            set_gate_start = 1.500
            set_gate_step = 0
            set_pul_wid = 500
            reset_vol_start = 1.0
            reset_vol_step = 0.1
            reset_gate_start = 4
            reset_gate_step = 0
            reset_pul_wid = 4000
        else:
            print('Unknown strategy! exit mapping process')
            return 0

        if poldirstr == 'POS':
            poldir = 'POS'
        elif poldirstr == 'NEG':
            poldir = 'NEG'
        else:
            poldir = 'POS'
            print('Error! invalid poldir! (assuming POS)', flush = True)

        set_vol_now = set_vol_start
        set_gate_now = set_gate_start
        reset_vol_now = reset_vol_start
        reset_gate_now = reset_gate_start

        last_operation = 0
        last_effective_operation_dir = 0
        map_succ_count = 0
        strongest_op_combo_count = 0
        op_interval_time = 0.000
        for idx in range(try_limit):

            result_adc_now = self.readOneCell(rowIdx, colIdx, poldirstr)

            if verbose:
                print('[%03d %03d %s]' % (rowIdx, colIdx, poldirstr), end = ' ')
            if abs(target_adc -
                   result_adc_now) <= tolerance:
                map_succ_count = map_succ_count + 1
                if verbose:
                    print('[%d] TAR: %d, CURR: %d, CNT: %d.' %
                          (idx, target_adc, result_adc_now, map_succ_count),
                          flush = True)

                last_operation = 0
                strongest_op_combo_count = 0

            elif target_adc > result_adc_now:
                map_succ_count = 0
                if last_operation == 1:
                    set_vol_now = min(set_vol_now + set_vol_step, set_vol_lim)
                    set_gate_now = min(set_gate_now + set_gate_step,
                                       set_gate_lim)
                elif last_effective_operation_dir == -1:
                    set_vol_now = set_vol_start
                    set_gate_now = set_gate_start
                else:
                    pass
                if verbose:
                    print('[%d] TAR: %d, CURR: %d, SET: %.2f V, GATE: %.2f V.' %
                          (idx, target_adc, result_adc_now, set_vol_now,
                           set_gate_now),
                          flush = True)

                self.setOneCell(rowIdx, colIdx, poldirstr, set_vol_now,
                                set_gate_now, set_pul_wid)

                if set_vol_now == set_vol_lim:
                    strongest_op_combo_count = strongest_op_combo_count + 1
                else:
                    strongest_op_count = 0

                last_operation = 1
                last_effective_operation_dir = 1

            else:
                map_succ_count = 0
                if last_operation == -1:
                    reset_vol_now = min(reset_vol_now + reset_vol_step,
                                        reset_vol_lim)
                    reset_gate_now = min(reset_gate_now + reset_gate_step,
                                         reset_gate_lim)
                elif last_effective_operation_dir == 1:
                    reset_vol_now = reset_vol_start
                    reset_gate_now = reset_gate_start
                else:
                    pass
                if verbose:
                    print('[%d] TAR: %d, CURR: %d, RESET: %.2f V, GATE: %.2f V.' %
                          (idx, target_adc, result_adc_now, reset_vol_now,
                           reset_gate_now),
                          flush = True)

                self.resetOneCell(rowIdx, colIdx, poldirstr, reset_vol_now,
                                  reset_gate_now, reset_pul_wid)

                if reset_vol_now == reset_vol_lim:
                    strongest_op_combo_count = strongest_op_combo_count + 1
                else:
                    strongest_op_combo_count = 0

                last_operation = -1
                last_effective_operation_dir = -1

            if map_succ_count >= 1:
                succ_flag = 1
                break
            if strongest_op_combo_count >= 10:
                break
        return succ_flag

    def setOneCell_1(self, rowIdx, colIdx, vbl, vwl, pulseWidth):
        row = rowIdx // 2
        if rowIdx % 2 == 0:
            POS_or_NEG = 'NEG'
        else:
            POS_or_NEG = 'POS'
        self.setOneCell(row, colIdx, POS_or_NEG, vbl, vwl, pulseWidth)

    def resetOneCell_1(self, rowIdx, colIdx, vsl, vwl, pulseWidth):
        row = rowIdx // 2
        if rowIdx % 2 == 0:
            POS_or_NEG = 'NEG'
        else:
            POS_or_NEG = 'POS'
        self.resetOneCell(row, colIdx, POS_or_NEG, vsl, vwl, pulseWidth)

    def formOneCell_1(self, rowIdx, colIdx, vbl, vwl, pulseWidth):
        row = rowIdx // 2
        if rowIdx % 2 == 0:
            POS_or_NEG = 'NEG'
        else:
            POS_or_NEG = 'POS'
        self.formOneCell(row, colIdx, POS_or_NEG, vbl, vwl, pulseWidth)

    def calcOneCell_1(self, rowIdx, colIdx):
        row = rowIdx // 2
        if rowIdx % 2 == 0:
            POS_or_NEG = 'NEG'
        else:
            POS_or_NEG = 'POS'
        return self.calcOneCell(row, colIdx, POS_or_NEG)

    def readOneCell_1(self, rowIdx, colIdx):
        row = rowIdx // 2
        if rowIdx % 2 == 0:
            POS_or_NEG = 'NEG'
        else:
            POS_or_NEG = 'POS'
        return self.readOneCell(row, colIdx, POS_or_NEG)

    def writeRDAC(self, numAd5254, channel, value):
        assert numAd5254 < 5
        assert channel < 4
        assert value <= 0xff
        if numAd5254 < 4:
            dev = AxiIIC(self.dictIic3)
        else:
            dev = AxiIIC(self.dictIic4)

        slaveAddr = 0x2c | (numAd5254 & 3)
        data = bytes([channel, value])
        dev.send(slaveAddr, data, 2)

    def cfgMapTimePara(self):
        self.cfgReadPulse(ti = 63, ts = 40, tb = 255, te = 60)

    def cfgCalcTimePara(self):
        self.cfgReadPulse(ti = 5, ts = 10, tb = 10, te = 10)

    def cfgProgPulseWidth(self, form_pul_wid = 100000, set_pul_wid = 500, reset_pul_wid = 4000):

        self.writeReg(REG57_ADDR, form_pul_wid // PULSE_UNIT_TIME)
        self.writeReg(REG58_ADDR, set_pul_wid // PULSE_UNIT_TIME)
        self.writeReg(REG59_ADDR, reset_pul_wid // PULSE_UNIT_TIME)

    def hw_adc_cal_switch(self, en):
        self.writeReg(HW_CAL_ADC_EN, en)

    def reg_init_config(self):
        self.writeReg(ROW_REARRANGE_EN, 3)
        self.writeReg(IRQ_PULSE_WIDTH, 500)
        self.hw_adc_cal_switch(0)

    def set_clac_adc_cfg(self, it_time, adc_intercept, weight_col_mean, colstart, colcount):
        self.hw_adc_cal_switch(1)
        self.writeReg(CAL_IT_TIME, it_time)
        for i in range(8):
            self.writeReg(ADC0_B_CALIB + i * 4, int(adc_intercept[i]))
        for i in range(128):
            if i >= colstart and i < colstart + colcount:
                self.writeBram(i * 4, int(weight_col_mean[i - colstart]))
            else:
                self.writeBram(i * 4, 0)

    def devInit(self):
        """Power on, initialize the device voltage and so on.

        Args:
            None

        Returns:
            None
        """

        self.switch_hd_version(0)
        self.resetSystem()
        self.cfgFloatBl()
        self.cfgClockMhz(CUR_CLOCK)
        self.cfgFRSPulse(ts = 0x50, tp = 0x3C, te = 0x3C)
        self.cfgReadPulse(ti = 63, ts = 40, tb = 255, te = 60)
        self.powerOn()
        self.cfgVBLMid(2.5)
        self.cfgVBLNeg(2.598)
        self.cfgVBLPos(2.399)
        self.cfgVrefCompDn(0)
        self.cfgVrefCompUp(4.699)
        self.cfgVSLClamp(2.5)
        self.cfgVDDLV(1.799)
        self.cfgSetVBL(2.814)
        self.cfgSetVWL(1.5)
        self.cfgResetVSL(4.298)
        self.cfgResetVWL(4.0)
        self.cfgFormVBL(4.556)
        self.cfgFormVWL(1.871)
        self.cfgProgPulseWidth()
        self.reg_init_config()

    def selectChip(self, chipNum):
        """Select the chip.

        Args:
            chipNum: int
                    Number of the selected chip, 0 <= chipNum < 12.

        Returns:
            None
        """
        assert chipNum < 12

        chipNum += 1
        self.writeReg(REG67_ADDR, chipNum)
        self.opFlow(OP_SELECT_CHIP)
        time.sleep(0.01)

    def readOneCell(self, rowIdx, colIdx, POS_or_NEG):
        """Read one cell.

        Args:
            rowIdx: int
                    Index of the selected chip, 0 <= rowIdx < 576.
            colIdx: int
                    Index of the selected chip, 0 <= colIdx < 128.
            POS_or_NEG: string
                    Select the location of the cell, 'POS' or 'NEG'.

        Returns:
            result: int
                    ADC value, 8 <= result < 16
        """

        self.oneCellOp(OP_READ, rowIdx, colIdx, POS_or_NEG)

        if POS_or_NEG == 'POS':
            value = self.readReg(REG76_ADDR)
        else:
            value = self.readReg(REG77_ADDR)

        adcVal = self.getAdcValue(value, colIdx)

        return adcVal

    def setOneCell(self, rowIdx, colIdx, POS_or_NEG, vbl, vwl, pulseWidth):
        """Set one cell.

        Args:
            rowIdx: int
                    Index of the selected chip, 0 <= rowIdx < 576.
            colIdx: int
                    Index of the selected chip, 0 <= colIdx < 128.
            POS_or_NEG: string
                    Select the location of the cell, 'POS' or 'NEG'.
            vbl: float
                    voltage of BL, unit(V)
            vwl: float
                    voltage of WL, unit(V)
            pulseWidth: int
                    Pulse width, unit(ns)

        Returns:
            None
        """
        self.cfgSetVBL(vbl)
        self.cfgSetVWL(vwl)
        self.writeReg(REG58_ADDR, pulseWidth // PULSE_UNIT_TIME)
        self.oneCellOp(OP_SET, rowIdx, colIdx, POS_or_NEG)

    def resetOneCell(self, rowIdx, colIdx, POS_or_NEG, vsl, vwl, pulseWidth):
        """Reset one cell.

        Args:
            rowIdx: int
                    Index of the selected chip, 0 <= rowIdx < 576.
            colIdx: int
                    Index of the selected chip, 0 <= colIdx < 128.
            POS_or_NEG: string
                    Select the location of the cell, 'POS' or 'NEG'.
            vsl: float
                    voltage of SL, unit(V)
            vwl: float
                    voltage of WL, unit(V)
            pulseWidth: int
                    Pulse width, unit(ns)

        Returns:
            None
        """

        self.cfgResetVSL(vsl)

        self.cfgResetVWL(vwl)

        self.writeReg(REG59_ADDR, pulseWidth // PULSE_UNIT_TIME)

        self.oneCellOp(OP_RESET, rowIdx, colIdx, POS_or_NEG)

    def formOneCell(self, rowIdx, colIdx, POS_or_NEG, vbl, vwl, pulseWidth):
        """Form one cell.

        Args:
            rowIdx: int
                    Index of the selected chip, 0 <= rowIdx < 576.
            colIdx: int
                    Index of the selected chip, 0 <= colIdx < 128.
            POS_or_NEG: string
                    Select the location of the cell, 'POS' or 'NEG'.
            vbl: float
                    voltage of BL, unit(V)
            vwl: float
                    voltage of WL, unit(V)
            pulseWidth: int
                    Pulse width, unit(ns)

        Returns:
            None
        """
        self.cfgFormVBL(vbl)
        self.cfgFormVWL(vwl)
        self.writeReg(REG57_ADDR, pulseWidth // PULSE_UNIT_TIME)
        self.oneCellOp(OP_FORM, rowIdx, colIdx, POS_or_NEG)

    def calcOneCell(self, rowIdx, colIdx, POS_or_NEG = 'POS'):
        """calculate one cell.

        Args:
            rowIdx: int
                    Index of the selected chip, 0 <= rowIdx < 576.
            colIdx: int
                    Index of the selected chip, 0 <= colIdx < 128.
            POS_or_NEG: string
                    Select the location of the cell, 'POS' or 'NEG'.

        Returns:
            result: int
                    ADC value, 0 <= result < 16
        """
        self.oneCellOp(OP_CALC, rowIdx, colIdx, POS_or_NEG)
        value = self.readReg(REG76_ADDR)
        adcVal = self.getAdcValue(value, colIdx)
        return adcVal

    def calcArray(self, rowInput: np.ndarray, rowStart, colStart, colCount):
        """calculate Selected cells.

        Args:
            rowInput: numpy.ndarray, two-axis matrix
                    The element of rowInput is 0, 1.
            colStart: int
                    Start column
            colCount: int
                    The count of selected column

        Returns:
            result: numpy.ndarray, two-axis matrix
                    The element of result is ADC value, 0 <= result < 16
        """
        assert (rowInput.dtype == 'uint8') or (rowInput.dtype == 'int8')

        bRowInput = bytes(rowInput)
        calcCount = int(rowInput.shape[0])
        rowCount = int(rowInput.shape[1])
        output = bytes(colCount * calcCount * [0])
        ret = self.clib.CalcArray_2(bRowInput, rowStart, rowCount, colStart,
                                    colCount, output, calcCount)
        if ret != 0:
            raise Exception('CalcArray_2() return error')
        output = np.frombuffer(output, dtype = np.uint8)
        output.resize(calcCount, colCount)

        return output

    def map_single_device_2T2R(self,
                               rowIdx,
                               colIdx,
                               target_adc,
                               tolerance = 0,
                               try_limit = 500,
                               strategy = 0,
                               with_form = 3,
                               verbose = 0):
        succ_flag = 0
        form_to_try_times = with_form

        set_vol_lim = 3
        set_gate_lim = 1.8
        reset_vol_lim = 3.5
        reset_gate_lim = 4.5

        op_interval_time = 0.000
        if not strategy:
            set_vol_start = 0.8
            set_vol_step = 0.05
            set_gate_start = 1.5
            set_gate_step = 0
            set_pul_wid = 500
            reset_vol_start = 0.6
            reset_vol_step = 0.05
            reset_gate_start = 4
            reset_gate_step = 0
            reset_pul_wid = 4000
        else:
            print('Unknown strategy! exit mapping process')
            return 0

        if verbose:
            print('\n\n')
            print('2T2R mapping start')
            print('target_adc: %d' % (target_adc))
            print('current state: %d' % (self.calcOneCell(rowIdx, colIdx, 'POS')), flush = True)

        if target_adc < 8:

            reset_succ = self.map_single_device(rowIdx,
                                                colIdx,
                                                'POS',
                                                8,
                                                tolerance = 0,
                                                verbose = verbose)
            if verbose:
                if reset_succ:
                    print('reset R_POS to HRS successfully')
                else:
                    print('reset R_POS to HRS failed')
                    return 0

            set_vol_now = set_vol_start
            set_gate_now = set_gate_start
            reset_vol_now = reset_vol_start
            reset_gate_now = reset_gate_start
            polstr = 'NEG'

            last_operation = 0
            last_effective_operation_dir = 0
            map_succ_count = 0
            strongest_op_combo_count = 0

            for idx in range(try_limit):

                result_adc_now = self.calcOneCell(rowIdx, colIdx, 'POS')

                if verbose:
                    print('[%03d %03d]' % (rowIdx, colIdx), end = ' ')
                    print('<R_NEG>', end = ' ')
                if abs(target_adc - result_adc_now) <= tolerance:
                    map_succ_count = map_succ_count + 1
                    if verbose:
                        print(
                            '[%d] TAR: %.1f, CURR: %d, CNT: %d.' %
                            (idx, target_adc, result_adc_now, map_succ_count),
                            flush = True)

                    last_operation = 0
                    strongest_op_combo_count = 0

                elif target_adc < result_adc_now:
                    map_succ_count = 0
                    if last_operation == 1:
                        set_vol_now = min(set_vol_now + set_vol_step,
                                          set_vol_lim)
                        set_gate_now = min(set_gate_now + set_gate_step,
                                           set_gate_lim)
                    elif last_effective_operation_dir == -1:
                        set_vol_now = set_vol_start
                        set_gate_now = set_gate_start
                    else:
                        pass
                    if verbose:
                        print('[%d] TAR: %.1f, CURR: %d, SET: %.2f V, GATE: %.2f V.'
                              % (idx, target_adc, result_adc_now, set_vol_now, set_gate_now),
                              flush = True)

                    self.setOneCell(rowIdx, colIdx, polstr, set_vol_now,
                                    set_gate_now, set_pul_wid)

                    if set_vol_now == set_vol_lim:
                        strongest_op_combo_count = strongest_op_combo_count + 1
                    else:
                        strongest_op_count = 0

                    last_operation = 1
                    last_effective_operation_dir = 1

                else:
                    map_succ_count = 0
                    if last_operation == -1:
                        reset_vol_now = min(reset_vol_now + reset_vol_step,
                                            reset_vol_lim)
                        reset_gate_now = min(reset_gate_now + reset_gate_step,
                                             reset_gate_lim)
                    elif last_effective_operation_dir == 1:
                        reset_vol_now = reset_vol_start
                        reset_gate_now = reset_gate_start
                    else:
                        pass
                    if verbose:
                        print(
                            '[%d] TAR: %.1f, CURR: %d, RESET: %.2f V, GATE: %.2f V.'
                            % (idx, target_adc, result_adc_now, reset_vol_now,
                               reset_gate_now),
                            flush = True)

                    self.resetOneCell(rowIdx, colIdx, polstr, reset_vol_now,
                                      reset_gate_now, reset_pul_wid)

                    if reset_vol_now == reset_vol_lim:
                        strongest_op_combo_count = strongest_op_combo_count + 1
                    else:
                        strongest_op_combo_count = 0

                    last_operation = -1
                    last_effective_operation_dir = -1

                if map_succ_count >= 1:
                    succ_flag = 1
                    break
                if strongest_op_combo_count >= 10:
                    if form_to_try_times > 0 and last_operation == 1:
                        form_to_try_times = form_to_try_times - 1
                        if verbose:
                            print('Try forming (%d times left)' %
                                  (form_to_try_times))
                        self.formOneCell(rowIdx, colIdx, 'NEG', 4.6, 2, 100000)
                        set_vol_now = set_vol_start
                        set_gate_now = set_gate_start
                        reset_vol_now = reset_vol_start
                        reset_gate_now = reset_gate_start
                        strongest_op_combo_count = 0
                        last_operation = 1
                        last_effective_operation_dir = 1
                    else:
                        break


        else:

            reset_succ = self.map_single_device(rowIdx,
                                                colIdx,
                                                'NEG',
                                                8,
                                                tolerance = 0,
                                                verbose = verbose)
            if verbose:
                if reset_succ:
                    print('reset R_NEG to HRS successfully')
                else:
                    print('reset R_NEG to HRS failed')
                    return 0

            set_vol_now = set_vol_start
            set_gate_now = set_gate_start
            reset_vol_now = reset_vol_start
            reset_gate_now = reset_gate_start
            polstr = 'POS'

            last_operation = 0
            last_effective_operation_dir = 0
            map_succ_count = 0
            strongest_op_combo_count = 0

            for idx in range(try_limit):
                result_adc_now = self.calcOneCell(rowIdx, colIdx, 'POS')

                if verbose:
                    print('[%03d %03d]' % (rowIdx, colIdx), end = ' ')
                    print('<R_POS>', end = ' ')
                if abs(target_adc -
                       result_adc_now) <= tolerance:
                    map_succ_count = map_succ_count + 1
                    if verbose:
                        print(
                            '[%d] TAR: %.1f, CURR: %d, CNT: %d.' %
                            (idx, target_adc, result_adc_now, map_succ_count),
                            flush = True)

                    last_operation = 0
                    strongest_op_combo_count = 0

                elif target_adc > result_adc_now:
                    map_succ_count = 0
                    if last_operation == 1:
                        set_vol_now = min(set_vol_now + set_vol_step,
                                          set_vol_lim)
                        set_gate_now = min(set_gate_now + set_gate_step,
                                           set_gate_lim)
                    elif last_effective_operation_dir == -1:
                        set_vol_now = set_vol_start
                        set_gate_now = set_gate_start
                    else:
                        pass
                    if verbose:
                        print('[%d] TAR: %.1f, CURR: %d, SET: %.2f V, GATE: %.2f V.'
                              % (idx, target_adc, result_adc_now, set_vol_now, set_gate_now),
                              flush = True)
                    self.setOneCell(rowIdx, colIdx, polstr, set_vol_now,
                                    set_gate_now, set_pul_wid)
                    if set_vol_now == set_vol_lim:
                        strongest_op_combo_count = strongest_op_combo_count + 1
                    else:
                        strongest_op_count = 0

                    last_operation = 1
                    last_effective_operation_dir = 1

                else:
                    map_succ_count = 0
                    if last_operation == -1:
                        reset_vol_now = min(reset_vol_now + reset_vol_step,
                                            reset_vol_lim)
                        reset_gate_now = min(reset_gate_now + reset_gate_step,
                                             reset_gate_lim)
                    elif last_effective_operation_dir == 1:
                        reset_vol_now = reset_vol_start
                        reset_gate_now = reset_gate_start
                    else:
                        pass
                    if verbose:
                        print(
                            '[%d] TAR: %.1f, CURR: %d, RESET: %.2f V, GATE: %.2f V.'
                            % (idx, target_adc, result_adc_now, reset_vol_now,
                               reset_gate_now),
                            flush = True)
                    self.resetOneCell(rowIdx, colIdx, polstr, reset_vol_now,
                                      reset_gate_now, reset_pul_wid)
                    if reset_vol_now == reset_vol_lim:
                        strongest_op_combo_count = strongest_op_combo_count + 1
                    else:
                        strongest_op_combo_count = 0

                    last_operation = -1
                    last_effective_operation_dir = -1

                if map_succ_count >= 1:
                    succ_flag = 1
                    break

                if strongest_op_combo_count >= 10:
                    if form_to_try_times > 0 and last_operation == 1:
                        form_to_try_times = form_to_try_times - 1
                        if verbose:
                            print('Try forming (%d times left)' %
                                  (form_to_try_times))
                        self.formOneCell(rowIdx, colIdx, 'POS', 4.6, 2, 100000)
                        set_vol_now = set_vol_start
                        set_gate_now = set_gate_start
                        reset_vol_now = reset_vol_start
                        reset_gate_now = reset_gate_start
                        strongest_op_combo_count = 0
                        last_operation = 1
                        last_effective_operation_dir = 1
                    else:
                        break

        if verbose:
            if succ_flag:
                print('mapping successful!', flush = True)
            else:
                print('mapping failed!', flush = True)
        return succ_flag

    def map_single_device_2T2R_POR(self,
                                   rowIdx,
                                   colIdx,
                                   target_adc,
                                   tolerance = 0,
                                   try_limit = 100,
                                   strategy = 0,
                                   with_form = 3,
                                   verbose = 0):
        succ_flag = 0
        form_to_try_times = with_form

        set_vol_lim = 3
        set_gate_lim = 1.8
        reset_vol_lim = 3.5
        reset_gate_lim = 4.5

        op_interval_time = 0.000
        if not strategy:
            set_vol_start = 0.8
            set_vol_step = 0.1
            set_gate_start = 1.5
            set_gate_step = 0
            set_pul_wid = 500
            reset_vol_start = 0.6
            reset_vol_step = 0.05
            orr_vol_start = 0.8
            orr_vol_step = 0.1
            reset_gate_start = 4
            reset_gate_step = 0
            orr_gate_step = 0
            reset_pul_wid = 4000
        else:
            print('Unknown strategy! exit mapping process')
            return 0

        if verbose:
            print('\n\n')
            print('2T2R mapping start')
            print('target_adc: %d' % (target_adc))
            print('current state: %d' % (self.calcOneCell(rowIdx, colIdx, 'POS')), flush = True)

        if target_adc < 8:

            reset_succ = self.map_single_device(rowIdx,
                                                colIdx,
                                                'POS',
                                                8,
                                                tolerance = 0,
                                                verbose = verbose)
            if verbose:
                if reset_succ:
                    print('reset R_POS to HRS successfully')
                else:
                    print('reset R_POS to HRS failed')
                    return 0

            set_vol_now = set_vol_start
            set_gate_now = set_gate_start
            reset_vol_now = reset_vol_start
            reset_gate_now = reset_gate_start
            polstr = 'NEG'

            last_operation = 0
            last_effective_operation_dir = 0
            map_succ_count = 0
            strongest_op_combo_count = 0

            for idx in range(try_limit):
                result_adc_now = self.calcOneCell(rowIdx, colIdx, 'POS')
                if verbose:
                    print('[%03d %03d]' % (rowIdx, colIdx), end = ' ')
                    print('<R_NEG>', end = ' ')
                if (abs(target_adc - result_adc_now) <= tolerance) and (
                        last_operation == 0 or last_operation == -1):
                    map_succ_count = map_succ_count + 1
                    if verbose:
                        print(
                            '[%d] TAR: %d, CURR: %d, CNT: %d, LAST_OP: %d.' %
                            (idx, target_adc, result_adc_now, map_succ_count, last_operation),
                            flush = True)

                    last_operation = 0
                    strongest_op_combo_count = 0
                    succ_flag = 1
                    break

                elif target_adc > result_adc_now + 1:
                    map_succ_count = 0
                    if last_operation == -2:
                        reset_vol_now = min(reset_vol_now + orr_vol_step,
                                            reset_vol_lim)
                        reset_gate_now = min(reset_gate_now + orr_gate_step,
                                             reset_gate_lim)

                    else:
                        reset_vol_now = orr_vol_start
                        reset_gate_now = reset_gate_start

                    if verbose:
                        print(
                            '[%d] TAR: %d, CURR: %d, RESET: %.2f V, GATE: %.2f V, LAST_OP: %d.'
                            % (idx, target_adc, result_adc_now, reset_vol_now,
                               reset_gate_now, last_operation),
                            flush = True)
                    self.resetOneCell(rowIdx, colIdx, polstr, reset_vol_now,
                                      reset_gate_now, reset_pul_wid)
                    if reset_vol_now == reset_vol_lim:
                        strongest_op_combo_count = strongest_op_combo_count + 1
                    else:
                        strongest_op_combo_count = 0

                    last_operation = -2
                    last_effective_operation_dir = -2

                elif (target_adc > result_adc_now) and (
                        last_operation == 0 or last_operation == -1):
                    map_succ_count = 0
                    if last_operation == -1:
                        reset_vol_now = min(reset_vol_now + reset_vol_step,
                                            reset_vol_lim)
                        reset_gate_now = min(reset_gate_now + reset_gate_step,
                                             reset_gate_lim)

                    else:
                        reset_vol_now = reset_vol_start
                        reset_gate_now = reset_gate_start

                    if verbose:
                        print(
                            '[%d] TAR: %d, CURR: %d, RESET: %.2f V, GATE: %.2f V, LAST_OP: %d.'
                            % (idx, target_adc, result_adc_now, reset_vol_now,
                               reset_gate_now, last_operation),
                            flush = True)
                    self.resetOneCell(rowIdx, colIdx, polstr, reset_vol_now,
                                      reset_gate_now, reset_pul_wid)
                    if reset_vol_now == reset_vol_lim:
                        strongest_op_combo_count = strongest_op_combo_count + 1
                    else:
                        strongest_op_combo_count = 0

                    last_operation = -1
                    last_effective_operation_dir = -1

                elif (target_adc > result_adc_now) and (
                        last_operation == 1 or last_operation == -2):
                    if verbose:
                        print(
                            '[%d] TAR: %d, CURR: %d, CNT: %d, LAST_OP: %d.' %
                            (idx, target_adc, result_adc_now, map_succ_count, last_operation),
                            flush = True)
                    last_operation = 0
                    strongest_op_combo_count = 0
                    succ_flag = 0
                    break

                else:
                    map_succ_count = 0
                    if last_operation == 1:
                        set_vol_now = min(set_vol_now + set_vol_step,
                                          set_vol_lim)
                        set_gate_now = min(set_gate_now + set_gate_step,
                                           set_gate_lim)

                    else:
                        set_vol_now = set_vol_start
                        set_gate_now = set_gate_start

                    if verbose:
                        print('[%d] TAR: %d, CURR: %d, SET: %.2f V, GATE: %.2f V, LAST_OP: %d.'
                              % (idx, target_adc, result_adc_now, set_vol_now, set_gate_now, last_operation),
                              flush = True)
                    self.setOneCell(rowIdx, colIdx, polstr, set_vol_now,
                                    set_gate_now, set_pul_wid)
                    if set_vol_now == set_vol_lim:
                        strongest_op_combo_count = strongest_op_combo_count + 1
                    else:
                        strongest_op_count = 0

                    last_operation = 1
                    last_effective_operation_dir = 1

                if strongest_op_combo_count >= 10:
                    if form_to_try_times > 0 and last_operation == 1:
                        form_to_try_times = form_to_try_times - 1
                        if verbose:
                            print('Try forming (%d times left)' %
                                  (form_to_try_times))
                        self.formOneCell(rowIdx, colIdx, 'NEG', 4.6, 2, 100000)
                        set_vol_now = set_vol_start
                        set_gate_now = set_gate_start
                        reset_vol_now = reset_vol_start
                        reset_gate_now = reset_gate_start
                        strongest_op_combo_count = 0
                        last_operation = 1
                        last_effective_operation_dir = 1
                    else:
                        break


        else:

            reset_succ = self.map_single_device(rowIdx,
                                                colIdx,
                                                'NEG',
                                                8,
                                                tolerance = 0,
                                                verbose = verbose)
            if verbose:
                if reset_succ:
                    print('reset R_NEG to HRS successfully')
                else:
                    print('reset R_NEG to HRS failed')
                    return 0

            set_vol_now = set_vol_start
            set_gate_now = set_gate_start
            reset_vol_now = reset_vol_start
            reset_gate_now = reset_gate_start
            polstr = 'POS'

            last_operation = 0
            last_effective_operation_dir = 0
            map_succ_count = 0
            strongest_op_combo_count = 0

            for idx in range(try_limit):
                result_adc_now = self.calcOneCell(rowIdx, colIdx, 'POS')
                if verbose:
                    print('[%03d %03d]' % (rowIdx, colIdx), end = ' ')
                    print('<R_POS>', end = ' ')
                if (abs(target_adc - result_adc_now) <= tolerance) and (
                        last_operation == 0 or last_operation == -1):
                    map_succ_count = map_succ_count + 1
                    if verbose:
                        print(
                            '[%d] TAR: %d, CURR: %d, CNT: %d, LAST_OP: %d.' %
                            (idx, target_adc, result_adc_now, map_succ_count, last_operation),
                            flush = True)

                    last_operation = 0
                    strongest_op_combo_count = 0
                    succ_flag = 1
                    break


                elif target_adc + 1 < result_adc_now:
                    map_succ_count = 0
                    if last_operation == -2:
                        reset_vol_now = min(reset_vol_now + orr_vol_step,
                                            reset_vol_lim)
                        reset_gate_now = min(reset_gate_now + orr_gate_step,
                                             reset_gate_lim)

                    else:
                        reset_vol_now = orr_vol_start
                        reset_gate_now = reset_gate_start

                    if verbose:
                        print(
                            '[%d] TAR: %d, CURR: %d, RESET: %.2f V, GATE: %.2f V, LAST_OP: %d.'
                            % (idx, target_adc, result_adc_now, reset_vol_now,
                               reset_gate_now, last_operation),
                            flush = True)
                    self.resetOneCell(rowIdx, colIdx, polstr, reset_vol_now,
                                      reset_gate_now, reset_pul_wid)
                    if reset_vol_now == reset_vol_lim:
                        strongest_op_combo_count = strongest_op_combo_count + 1
                    else:
                        strongest_op_combo_count = 0

                    last_operation = -2
                    last_effective_operation_dir = -1

                elif (target_adc < result_adc_now) and (
                        last_operation == 0 or last_operation == -1):
                    map_succ_count = 0
                    if last_operation == -1:
                        reset_vol_now = min(reset_vol_now + reset_vol_step,
                                            reset_vol_lim)
                        reset_gate_now = min(reset_gate_now + reset_gate_step,
                                             reset_gate_lim)

                    else:
                        reset_vol_now = reset_vol_start
                        reset_gate_now = reset_gate_start

                    if verbose:
                        print(
                            '[%d] TAR: %d, CURR: %d, RESET: %.2f V, GATE: %.2f V, LAST_OP: %d.'
                            % (idx, target_adc, result_adc_now, reset_vol_now,
                               reset_gate_now, last_operation),
                            flush = True)
                    self.resetOneCell(rowIdx, colIdx, polstr, reset_vol_now,
                                      reset_gate_now, reset_pul_wid)
                    if reset_vol_now == reset_vol_lim:
                        strongest_op_combo_count = strongest_op_combo_count + 1
                    else:
                        strongest_op_combo_count = 0

                    last_operation = -1
                    last_effective_operation_dir = -1

                elif (target_adc < result_adc_now) and (
                        last_operation == 1 or last_operation == -2):
                    if verbose:
                        print(
                            '[%d] TAR: %d, CURR: %d, CNT: %d, LAST_OP: %d.' %
                            (idx, target_adc, result_adc_now, map_succ_count, last_operation),
                            flush = True)
                    last_operation = 0
                    strongest_op_combo_count = 0
                    succ_flag = 0
                    break


                else:
                    map_succ_count = 0
                    if last_operation == 1:
                        set_vol_now = min(set_vol_now + set_vol_step,
                                          set_vol_lim)
                        set_gate_now = min(set_gate_now + set_gate_step,
                                           set_gate_lim)

                    else:
                        set_vol_now = set_vol_start
                        set_gate_now = set_gate_start

                    if verbose:
                        print('[%d] TAR: %d, CURR: %d, SET: %.2f V, GATE: %.2f V, LAST_OP: %d.'
                              % (idx, target_adc, result_adc_now, set_vol_now, set_gate_now, last_operation),
                              flush = True)
                    self.setOneCell(rowIdx, colIdx, polstr, set_vol_now,
                                    set_gate_now, set_pul_wid)
                    if set_vol_now == set_vol_lim:
                        strongest_op_combo_count = strongest_op_combo_count + 1
                    else:
                        strongest_op_count = 0

                    last_operation = 1
                    last_effective_operation_dir = 1

                if strongest_op_combo_count >= 10:
                    if form_to_try_times > 0 and last_operation == 1:
                        form_to_try_times = form_to_try_times - 1
                        if verbose:
                            print('Try forming (%d times left)' %
                                  (form_to_try_times))
                        self.formOneCell(rowIdx, colIdx, 'POS', 4.6, 2, 100000)
                        set_vol_now = set_vol_start
                        set_gate_now = set_gate_start
                        reset_vol_now = reset_vol_start
                        reset_gate_now = reset_gate_start
                        strongest_op_combo_count = 0
                        last_operation = 1
                        last_effective_operation_dir = 1
                    else:
                        break

        if verbose:
            if succ_flag:
                print('mapping successful!', flush = True)
            else:
                print('mapping failed!', flush = True)
        return succ_flag

    def calc_array(self, addr, input):
        """calculate Selected cells.

        Args:
            addr: Addrinfo
                    Selected area
            input:  input data for rowselect

        Returns:
            result: list
                    ADC values of selected area
        """

        rowstart, rowcount, colstart, colcount = addr
        time1 = time.time()
        output = self.calcArray(input, rowstart, colstart, colcount)
        time2 = time.time()

        return output

    def elemem_set_read_mask(self, addr, maskInput):
        """set read weight mask.

        Args:
             rowCount: 掩码区域行高

        Returns:
            output: weight
        """

        rowStart, colStart, rowCount, colCount = addr
        assert (maskInput.dtype == 'uint8') or (maskInput.dtype == 'int8')
        assert (rowCount == maskInput.shape[0])

        read_mask = np.full(shape = (rowCount, 128), fill_value = 0, dtype = np.uint8)
        read_mask[rowStart:rowStart + rowCount, colStart:colStart + colCount] = maskInput

        bread_mask = bytes(read_mask)

        ret = self.clib.ElememDev_SetReadMask(rowCount, bread_mask, len(bread_mask))
        if ret != 0:
            raise Exception('ElememDev_SetReadMask() return error')

        return True

    def elemem_read_weight(self, rowStart, rowCount, colStart, colCount, type_is_1t1r = 0, acc_times = 1, with_mask = 0, time_out_ms = 5 * 1000):
        """read weight.

        Args:
            rowStart, rowCount, colStart, colCount: 读取区域坐标
            type_is_1t1r: 读取数据类型是否为1t1r, 否则为2t2r
            time_out_ms: 读取接口阻塞超时时间

        Returns:
            output: weight
        """
        if type_is_1t1r:
            output = bytes(rowCount * colCount * [0] * 2)
        else:
            output = bytes(rowCount * colCount * [0])
        ret = self.clib.ElememDev_ReadWeight(rowStart, colStart, rowCount, colCount, output, type_is_1t1r, acc_times, with_mask, time_out_ms)
        if ret != 0:
            raise Exception('ElememDev_ReadWeight() return error')
        output = np.frombuffer(output, dtype = np.uint8)
        output = output.reshape(-1, colCount)

        return output

    def elemem_block_write_weight(self, weightInput: np.ndarray, rowStart, colStart, prog_cycle = 40):
        '''
        write weight.

        Returns:
            result: pass cell number
        '''
        assert (weightInput.dtype == 'uint8') or (weightInput.dtype == 'int8')
        data_is_zero = 0
        bweightInput = bytes(weightInput)
        rowCount = int(weightInput.shape[0])
        colCount = int(weightInput.shape[1])
        pass_num_arr = bytes(prog_cycle * [0] * 4)
        ret_pass_num = self.clib.ElememDev_BlockProgram(rowStart, colStart, rowCount, colCount, data_is_zero, bweightInput, prog_cycle, pass_num_arr)
        if ret_pass_num < 0:
            raise Exception('ElememDev_BlockProgram() return error:', ret_pass_num)
        pass_num_arr = np.frombuffer(pass_num_arr, dtype = np.int32, count = prog_cycle)
        return ret_pass_num, pass_num_arr

    def elemem_block_write_debug_log(self, addr, buf_select = 0, csv_log = False):
        """
        Set/reset operation count
        """
        rowCount = addr[2]
        colCount = addr[3]
        output = bytes(rowCount * colCount * [0])
        self.clib.ElememDev_ReadDebugBuf(output, rowCount * colCount, buf_select)
        output = np.frombuffer(output, dtype = np.uint8)
        output.resize(rowCount, colCount)
        return output

    def elemem_calc_array(self, rowInput: np.ndarray, rowStart, rowCount, colStart, colCount,
                          data_type = 1, split_mode = 0, ret_mode = 0, time_out_ms = 1000):
        """calculate Selected cells.

        Args:
            rowInput: numpy.ndarray, two-axis matrix
            data_type: int
                    input data type, 1:int2; 2:int3,...,7:int8
            split_mode: int
                    data expand mode, 0:全展开; 1:位展开

        Returns:
            result: numpy.ndarray, two-axis matrix
                    The element of result(int16)
        """
        assert (rowInput.dtype == 'uint8') or (rowInput.dtype == 'int8')
        bRowInput = bytes(rowInput)
        calcCount = int(rowInput.shape[0])
        rowCount = int(rowInput.shape[1])
        if ret_mode:
            if split_mode:
                output_len = colCount * calcCount * 4 * 8
            else:
                output_len = colCount * calcCount * 4 * 128
            if output_len > 327680000:
                raise Exception('calc output data size too large')
        else:
            output_len = colCount * calcCount * 4
        output = bytes(output_len * [0])

        ret_len = self.clib.ElememDev_CalcArray(bRowInput, rowStart, rowCount, colStart, colCount,
                                                output, calcCount, data_type, split_mode, ret_mode, time_out_ms)

        if ret_len <= 0:
            raise Exception('elemem_calc_array() return error')

        output = np.frombuffer(output, dtype = np.int32, count = ret_len)
        if ret_mode:
            output.resize(ret_len // colCount, colCount)
        else:
            output.resize(calcCount, colCount)

        return output
