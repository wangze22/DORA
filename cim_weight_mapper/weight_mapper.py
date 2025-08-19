'''
@author: Wang Ze
'''
import colorsys
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def map_blocks_to_boxes(block_dict, box_size):
    def is_larger(A, B):

        rows_A, cols_A = A.shape
        rows_B, cols_B = B.shape

        return rows_A > rows_B or cols_A > cols_B

    def convert_matrix_to_tuples(matrix):
        return [(val, i) for i, val in enumerate(matrix)]

    def find_left_most_col_for_row(search_points, target_row, start_col):
        while start_col > 0:

            for point in search_points:
                if point[0] < target_row and point[1] == start_col - 1:
                    start_col -= 1
                    break
            else:

                break
        return start_col

    def add_array(array_info):
        idx = len(array_info)
        array_info[idx] = {}
        array_info[idx]['array_data'] = torch.zeros(box_size)
        array_info[idx]['array_usage'] = np.zeros(box_size, dtype = bool)
        array_info[idx]['height_map'] = np.zeros(box_size[1], dtype = int)
        array_info[idx]['search_points'] = [(0, 0)]
        array_info[idx]['has_data'] = False

    def find_first_col(height_map):
        search_points = convert_matrix_to_tuples(height_map)
        search_points.sort(key = lambda x: (x[0], x[1]))

        new_values = []
        continuous_start_col = None
        previous_row = None
        previous_col = None

        for point in search_points:
            row, col = point

            if previous_row is None or row != previous_row:
                if continuous_start_col is not None:
                    adjusted_start_col = find_left_most_col_for_row(search_points, previous_row,
                                                                    continuous_start_col)
                    new_values.append((previous_row, adjusted_start_col))
                continuous_start_col = col

            elif col != previous_col + 1:

                adjusted_start_col = find_left_most_col_for_row(search_points, previous_row,
                                                                continuous_start_col)
                new_values.append((previous_row, adjusted_start_col))
                continuous_start_col = col
            previous_row, previous_col = point

        if continuous_start_col is not None:
            adjusted_start_col = find_left_most_col_for_row(search_points, previous_row, continuous_start_col)
            new_values.append((previous_row, adjusted_start_col))

        return new_values

    if len(block_dict) == 0:
        print(f'No Weight For Mapping')
        return
    array_data = {}
    array_usage = {}
    search_points = {}
    height_map = {}

    for key, block in block_dict.items():
        block['placed'] = False

    array_info = {}

    add_array(array_info)

    array_idx = 0

    block_mapping_info = {}

    while not all(block['placed'] for block_key, block in block_dict.items()):

        full_search = True
        search_points = array_info[array_idx]['search_points']
        array_data = array_info[array_idx]['array_data']
        array_usage = array_info[array_idx]['array_usage']
        height_map = array_info[array_idx]['height_map']

        for point in search_points:
            (row, col) = point
            for block_key, block in block_dict.items():
                placed_info = block['placed']
                if placed_info:
                    continue
                placed = placed_info
                block_data = block['data']

                if is_larger(block_data, array_data):
                    print(f'Weight is Larger than Array. Unable to map.')
                    exit(1)

                is_row_in_range = row + block_data.shape[0] <= array_data.shape[0]

                is_col_in_range = col + block_data.shape[1] <= array_data.shape[1]

                start_row, end_row = row, row + block_data.shape[0]
                start_col, end_col = col, col + block_data.shape[1]

                is_area_unused = not np.any(array_usage[start_row:end_row, start_col:end_col])

                if is_row_in_range and is_col_in_range and is_area_unused:
                    target_array_section = array_data[start_row:end_row, start_col:end_col]
                    target_array_section[:] = block_data

                    usage_section = array_usage[start_row:end_row, start_col:end_col]
                    usage_section[:] = True

                    start_col = col
                    end_col = col + block_data.shape[1]

                    height_map[start_col:end_col] = np.maximum(
                        height_map[start_col:end_col],
                        row + block_data.shape[0])

                    block_mapping_info[block_key] = {}
                    block_mapping_info[block_key]['weight_addr'] = (
                        int(row), int(col), int(block_data.shape[0]), int(block_data.shape[1]))

                    block_mapping_info[block_key]['array_idx'] = array_idx

                    placed = True
                    block['placed'] = placed

                    search_points = find_first_col(height_map)
                    array_info[array_idx]['search_points'] = search_points
                    array_info[array_idx]['has_data'] = True

                    break

            if placed:
                full_search = False
                break

        if full_search or all(block['placed'] for _, block in block_dict.items()):
            add_array(array_info)
            array_idx += 1

    return block_mapping_info
